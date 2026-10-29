/**
 *  @file include/usearch/dense/vram_nvidia_compaction.hpp
 *  @author Ash Vardanian
 *  @date October 22, 2026
 *  @brief The freestanding compaction of @c vram_nvidia_indexes sources into a RAM target under a
 *      shard capacity.
 *
 *  A compaction packs the source shards largest first into target shards of at most a shard
 *  capacity of rows, each into the first with room, so it ends with as few shards as fit. The
 *  shards of one source landing side by side in one target shard form fold bins, which link on the
 *  GPUs before anything crosses: every row searches the bin's other shards through the normal
 *  search, re-selects its neighbors from its own and those over the whole bin as one storage, and
 *  every pick in another shard merges back into the picked row. Each bin then crosses to the host
 *  through its shards' save plans and lands in its target shard through one load, its edges in the
 *  bin's dense ID space and its keys beside it, and each target shard heals.
 *
 *  The one header that sees both residencies: it includes the NVIDIA tower and the portable RAM
 *  one, which never include each other.
 */
#pragma once

#include <cstring> // `std::memset` - the load cursors between bins

#include <bit>         // `std::bit_width` - a bin's shard bits
#include <bitset>      // `std::bitset` - the devices a bin's fold buffers map for
#include <span>        // `std::span`
#include <type_traits> // `std::is_same_v`
#include <utility>     // `std::exchange`

#include "usearch/dense/types.hpp" // `compact_config_t`, `compact_result_t`
#include "usearch/dense/ram_indexes.hpp"
#include "usearch/dense/vram_nvidia_indexes.hpp"

namespace unum::usearch::dense {

/**
 *  @brief Opens a compaction of the device @p sources into the host @p target, each target shard
 *      holding at most @p config.shard_capacity rows.
 *
 *  The target and every source close here, each under its own gate, so the packing never goes
 *  stale: the target stays closed while the plan lives and each source until it lands, and
 *  insertions into any of them answer @c contended_k meanwhile. Searches of the sources run
 *  throughout, since the fold writes only its own buffers, retrying once around the window that
 *  empties each; searches of the target retry around the commits that publish each source's rows.
 *  Every row moves, so the sources end empty, closed and sealed, keeping their shards and storage,
 *  and their slots must outlive the plan.
 *
 *  @param[in] sources The device indexes giving their rows up, on the devices @p device_executor
 *      drives.
 *  @param[in,out] target The host index receiving them; one never reserved takes the sources'
 *      geometry.
 *  @param[in,out] device_executor The streams every source shard folds and saves through.
 *  @param[in,out] host_executor The threads every target shard loads and heals on.
 *  @param[in] config The shard capacity every target shard holds to.
 *
 *  @par Statuses:
 *  - @c uninitialized_k No source is named, the target has no shards, or @p host_executor has no
 *    threads; or a source keeps it, as its @c status says.
 *  - @c config_mismatch_k A null or repeated source, another metric or geometry, a source device
 *    @p device_executor drives no stream on, or a bin's device that cannot reach another
 *    member's storage; or a source keeps it.
 *  - @c contended_k A call or batch in flight on the target or a source, or another refine,
 *    absorb, split, compaction, save or load holding either.
 *  - @c unsupported_k A source shard past the capacity, which would have to be cut first, or a
 *    source keeps it.
 *  - @c overflow_k A fresh target shard generation still has no room for a source shard, or a
 *    source keeps it.
 *  - @c cancelled_k or @c unsupported_layout_k A source keeps it, as its @c status says.
 */
template <typename vram_indexes_type_, typename ram_indexes_type_>
expected<vram_nvidia_compact_plan<vram_indexes_type_, ram_indexes_type_>> plan_compact(
    std::span<vram_indexes_type_* const> sources, ram_indexes_type_& target, cuda_executor_t& device_executor,
    typename ram_indexes_type_::executor_t& host_executor, compact_config_t const& config = {}) noexcept;

/**
 *  The one-shot @c plan_compact, run and finished on contexts carved for the call; @p on_visited
 *  reports and cancels the fold alone, as in @c vram_nvidia_compact_plan::run.
 *
 *  @par Statuses:
 *  - @c uninitialized_k No source is named, the target has no shards, or @p host_executor has no
 *    threads; or a source keeps it, as its @c status says.
 *  - @c config_mismatch_k A null or repeated source, another metric or geometry, a source device
 *    @p device_executor drives no stream on, a bin's device that cannot reach another member's
 *    storage, or a source's rows that never land; or a source keeps it.
 *  - @c contended_k A call or batch in flight on the target or a source, another refine, absorb,
 *    split, compaction, save or load holding either, or another call holding an executor.
 *  - @c unsupported_k A source shard past the capacity, which would have to be cut first, or a
 *    source keeps it.
 *  - @c overflow_k A fresh target shard generation still has no room for a source shard, the
 *    landed rows pass a target shard's ceiling or key column or the locator chunks, or a source
 *    keeps it.
 *  - @c unsupported_layout_k A source keeps it, as its @c status says.
 */
template <typename vram_indexes_type_, typename ram_indexes_type_, typename progress_type_ = dummy_progress_t>
expected<compact_result_t> compact(std::span<vram_indexes_type_* const> sources, ram_indexes_type_& target,
                                   cuda_executor_t& device_executor,
                                   typename ram_indexes_type_::executor_t& host_executor,
                                   compact_config_t const& config = {}, progress_type_&& on_visited = {}) noexcept;

/**
 *  @brief One compaction in flight from NVIDIA device shards into a RAM target, holding every
 *      index it moves closed meanwhile.
 *
 *  @ref run drives the fold's three phases on the devices - the cross searches, the re-selection
 *  and the merge - then lands the sources still pending, each bin staged on the host once through
 *  columns the plan sized at planning for the widest; a cancel pauses between phases, and a later
 *  call continues there. @ref finish heals every target shard once the sources landed, reopens
 *  the target and every source still pending, and reports.
 *
 *  A bin of @b S shards spends the top @c bit_width(S-1) bits of a bin ID on a shard's place and
 *  the rest on its row, so a shard joins a bin only while its rows fit those, and one too large
 *  starts a bin of its own, as does one that would leave each other member less than one slot of a
 *  row. Every row holds @c cross_count matches from each other shard, one row of newcomers split
 *  evenly, which the re-selection weighs beside its live members, as a back-link round caches a row
 *  of newcomers beside the row it holds.
 */
template <typename vram_indexes_type_, typename ram_indexes_type_>
struct [[nodiscard]] vram_nvidia_compact_plan {
    using vram_indexes_t = vram_indexes_type_;
    using ram_indexes_t = ram_indexes_type_;
    using device_executor_t = typename vram_indexes_t::executor_t;
    using host_executor_t = typename ram_indexes_t::executor_t;
    using traversal_context_t = typename ram_indexes_t::traversal_context_t;
    using refine_plan_t = typename ram_indexes_t::refine_plan_t;
    using result_t = compact_result_t;
    static_assert(vram_indexes_t::vector_dtype_k == ram_indexes_t::vector_dtype_k,
                  "Rows cross as stored, so both residencies keep one scalar");
    static_assert(std::is_same_v<typename vram_indexes_t::key_t, typename ram_indexes_t::key_t>,
                  "Keys cross as stored, so both residencies keep one key");

    vram_nvidia_compact_plan() noexcept = default;
    vram_nvidia_compact_plan(vram_nvidia_compact_plan&& other) noexcept
        : refining_(std::move(other.refining_)), columns_(std::move(other.columns_)),
          members_(std::move(other.members_)), bins_(std::move(other.bins_)), folds_(std::move(other.folds_)),
          bin_tables_(std::move(other.bin_tables_)), fold_tables_(std::move(other.fold_tables_)),
          arenas_(std::move(other.arenas_)), walks_(std::move(other.walks_)),
          sources_(std::exchange(other.sources_, {})), landed_(other.landed_), moved_entries_(other.moved_entries_),
          folded_(other.folded_), phase_(other.phase_) {}
    vram_nvidia_compact_plan& operator=(vram_nvidia_compact_plan&&) = delete;
    vram_nvidia_compact_plan(vram_nvidia_compact_plan const&) = delete;
    vram_nvidia_compact_plan& operator=(vram_nvidia_compact_plan const&) = delete;

    /** Reopens every source still pending; the refine plan reopens the target. */
    ~vram_nvidia_compact_plan() noexcept { release_sources_(); }

    /**
     *  @brief Drives the fold's pending phases, then lands the pending sources; @c cancelled_k
     *      means a later call continues at the next phase.
     *
     *  A source failing to land stays pending and closed, with every later one, for the next call.
     *
     *  @param[in,out] device_executor The streams the plan was opened over.
     *  @param[in,out] host_executor The threads the plan was opened over.
     *  @param[in,out] traversals The caller's context of every host thread, which @ref finish heals
     *      on; the call checks their count.
     *  @param[in] on_visited Fires once the re-selection and once the merge landed, with the rows
     *      folded so far and the rows moving; a @c cancel_k answer stops before the next phase.
     *
     *  @par Statuses:
     *  - @c uninitialized_k The plan is empty or finished.
     *  - @c config_mismatch_k The @p traversals are not one per host thread, or a source's rows
     *    never land.
     *  - @c contended_k Another call holds either executor or a source's gate, or a target shard
     *    has a batch or load in flight.
     *  - @c overflow_k The landed rows pass a target shard's ceiling or key column, or the locator
     *    chunks.
     */
    template <typename progress_type_ = dummy_progress_t>
    status_t run(device_executor_t& device_executor, host_executor_t& host_executor,
                 std::span<traversal_context_t> traversals, progress_type_&& on_visited = {}) noexcept {
        ram_indexes_t* const target = refining_.owner_;
        if (!target) return status_t::uninitialized_k;
        if (host_executor.size() != traversals.size()) return status_t::config_mismatch_k;
        auto [device_claim, device_status] = device_executor.claim();
        if (failed(device_status)) return device_status;
        auto [host_claim, host_status] = host_executor.claim();
        if (failed(host_status)) return host_status;
        if (status_t const folded = fold_(on_visited); failed(folded)) return folded;
        if (status_t const landed = land_pending_(*target, device_executor, host_executor); failed(landed))
            return landed;
        // The fold linked every bin, so each target shard's pass only heals, which `finish` runs.
        if (refining_.shard_plans_.empty()) return refining_.open_shards_();
        return status_t::success_k;
    }

    /**
     *  Heals on the caller's @p traversals, one per host thread, reopens the target and reports. A
     *  plan whose sources have not all landed heals nothing.
     *
     *  @par Statuses:
     *  - @c uninitialized_k The plan is empty or finished, or @p host_executor holds no pool or no
     *    workers.
     *  - @c config_mismatch_k The @p traversals hold other than one context per thread.
     *  - @c contended_k Another call holds @p host_executor, or its pool runs another dispatch.
     *  - @c overflow_k A queue capacity wraps when rounded.
     */
    expected<result_t> finish(host_executor_t& host_executor, std::span<traversal_context_t> traversals) noexcept {
        result_t result;
        ram_indexes_t* const target = refining_.owner_;
        if (!target) return {result, status_t::uninitialized_k};
        result.source_shards = refining_.landings_.size();
        result.moved_entries = moved_entries_;
        release_sources_();
        auto [graph, status] = refining_.finish(host_executor, traversals);
        result.graph = graph;
        result.graph += folded_;
        for (std::size_t shard = 0; shard != target->shards_count(); ++shard)
            if (target->shard(shard).size()) ++result.target_shards;
        return {result, status};
    }

  private:
    template <typename other_vram_type_, typename other_ram_type_>
    friend expected<vram_nvidia_compact_plan<other_vram_type_, other_ram_type_>> plan_compact(
        std::span<other_vram_type_* const>, other_ram_type_&, cuda_executor_t&, typename other_ram_type_::executor_t&,
        compact_config_t const&) noexcept;
    template <typename other_vram_type_, typename other_ram_type_, typename progress_type_>
    friend expected<compact_result_t> compact(std::span<other_vram_type_* const>, other_ram_type_&, cuda_executor_t&,
                                              typename other_ram_type_::executor_t&, compact_config_t const&,
                                              progress_type_&&) noexcept;

    using vram_index_t = typename vram_indexes_t::index_t;
    using vram_shard_t = typename vram_indexes_t::shard_t;
    using fanout_t = typename vram_indexes_t::fanout_t;
    using walk_context_t = typename vram_indexes_t::traversal_context_t;
    using search_config_t = typename vram_index_t::search_config_t;
    using ram_index_t = typename ram_indexes_t::index_t;
    using ram_shard_t = typename ram_indexes_t::shard_t;
    using ram_load_plan_t = typename ram_index_t::load_plan_t;

    /** A host shard and a device shard never share a placement, so every target shard ranks as
     *  another placement's. */
    struct placement_of_t {
        bool operator()(ram_shard_t const&) const noexcept { return false; }
        bool operator()(vram_shard_t const&) const noexcept { return true; }
    };

    /** One bin: the run of @c members_ from @c first, shards of one source landing side by side in
     *  one target shard in their landing order, and the first of its members' fold states, which
     *  only a folding bin holds. */
    struct bin_t {
        std::size_t first = 0;
        std::size_t count = 0;
        std::size_t state = 0;
    };

    /** The fold's phases, the last one landing what it linked. */
    enum class phase_t : u8_t { cross_k, reselect_k, merge_k, land_k };

    explicit vram_nvidia_compact_plan(forkunion_allocator_t allocator) noexcept
        : refining_(allocator), columns_(allocator), members_(allocator), bins_(allocator), folds_(allocator),
          bin_tables_(allocator), fold_tables_(allocator), arenas_(allocator), walks_(allocator) {}

    /** Bytes one row of @p index takes in the host columns: its packed vector, then its
     *  neighbors. */
    static std::size_t staged_row_bytes_(vram_index_t const& index) noexcept {
        return packed_bytes(vram_indexes_t::vector_dtype_k, index.dimensions()) +
               index.connectivity() * sizeof(node_id_t);
    }

    /** Low bits a bin ID of a bin of @p shards shards, two at least, keeps a shard's row in. */
    static u32_t row_bits_(std::size_t shards) noexcept { return 32u - static_cast<u32_t>(std::bit_width(shards - 1)); }

    /** Matches a row of a bin of @p shards shards holds from each other shard: a row of
     *  @p connectivity newcomers split evenly, which @c cut_bins_ keeps at one at least, so the
     *  cross results fit a row and a fold warp caches what a re-selection does. */
    static u32_t cross_count_(std::size_t shards, std::size_t connectivity) noexcept {
        return static_cast<u32_t>(connectivity / (shards - 1));
    }

    /** A source shard as a plural and its place in that plural's shards. */
    struct located_shard_t {
        vram_indexes_t* indexes = nullptr;
        std::size_t shard = 0;
    };

    /** The source holding the source shard @p global numbers across the sources in order, and the
     *  shard's place in it. */
    located_shard_t locate_(std::size_t global) const noexcept {
        std::size_t source = 0;
        for (; global >= sources_[source]->shards_.size(); ++source) global -= sources_[source]->shards_.size();
        return {sources_[source], global};
    }

    /** The source shard @p global numbers across the sources in order. */
    vram_shard_t& shard_at_(std::size_t global) const noexcept {
        located_shard_t const located = locate_(global);
        return located.indexes->shards_[located.shard];
    }

    /** The executor's stream on the device of the source shard @p global numbers. */
    typename device_executor_t::stream_t& stream_at_(device_executor_t& executor, std::size_t global) const noexcept {
        located_shard_t const located = locate_(global);
        return executor.stream(fanout_t::stream_of(located.indexes->shards_, executor, located.shard));
    }

    /** Rows the members of @p bin hold together. */
    std::size_t bin_rows_(bin_t const& bin) const noexcept {
        std::size_t rows = 0;
        for (std::size_t member = bin.first; member != bin.first + bin.count; ++member)
            rows += shard_at_(members_[member]).index->size();
        return rows;
    }

    /** Validates @p sources against @p target, closes the target and every source, each under its
     *  own gate, packs every source shard into the target under the tighter of
     *  @p config.shard_capacity and the target's own bound, cuts the landings into bins, and sizes
     *  the host columns for the widest bin and every folding bin's buffers and contexts. */
    static expected<vram_nvidia_compact_plan> open_(std::span<vram_indexes_t* const> sources, ram_indexes_t& target,
                                                    device_executor_t& device_executor,
                                                    host_executor_t const& host_executor,
                                                    compact_config_t const& config) noexcept {
        if (sources.empty() || !target.shards_count() || !host_executor.size()) return {{}, status_t::uninitialized_k};
        vram_indexes_t const* shaped = sources.front();
        for (std::size_t position = 0; position != sources.size(); ++position) {
            vram_indexes_t const* source = sources[position];
            if (!source || source->kernel() != target.kernel()) return {{}, status_t::config_mismatch_k};
            for (std::size_t earlier = 0; earlier != position; ++earlier)
                if (sources[earlier] == source) return {{}, status_t::config_mismatch_k};
            if (status_t const kept = source->status(); failed(kept)) return {{}, kept};
            if (status_t const paired = fanout_t::pairs_with(source->shards_, device_executor); failed(paired))
                return {{}, paired};
            if (!source->capacity()) continue;
            if (!shaped->capacity()) shaped = source;
            if (source->dimensions() != shaped->dimensions() || source->connectivity() != shaped->connectivity() ||
                (target.capacity() &&
                 (source->dimensions() != target.dimensions() || source->connectivity() != target.connectivity())))
                return {{}, status_t::config_mismatch_k};
        }
        // Zero, or anything past the target's own bound, means that bound.
        std::size_t const capacity = config.shard_capacity && config.shard_capacity < target.shard_capacity()
                                         ? config.shard_capacity
                                         : target.shard_capacity();

        vram_nvidia_compact_plan plan {target.shards_.allocator()};
        refine_plan_t& refining = plan.refining_;
        if (status_t const closed = target.close_gated_(); failed(closed)) return {{}, closed};
        refining.owner_ = &target;
        refining.config_.expansion_factor = config.expansion_factor;
        for (std::size_t closing = 0; closing != sources.size(); ++closing)
            if (status_t const closed = sources[closing]->close_gated_(); failed(closed)) {
                while (closing) sources[--closing]->shards_.reopen();
                return {{}, closed};
            }
        plan.sources_ = sources;

        // Closed, the sources hold still, so their shards are what the landings pack.
        std::size_t count = 0;
        for (vram_indexes_t const* source : sources) {
            count += source->shards_.size();
            plan.moved_entries_ += source->size();
        }
        if (failed(refining.landings_.resize(count))) return {{}, status_t::bad_alloc_k};
        auto source_shard = [&](std::size_t from) noexcept -> vram_shard_t const& { return plan.shard_at_(from); };
        // Only a source with rows needs a shard generation, so the geometry to take exists.
        auto open = [&]() noexcept {
            if (!target.capacity())
                if (status_t const reserved =
                        target.reserve({0, static_cast<connectivity_t>(shaped->connectivity()), shaped->dimensions()})
                            .status;
                    failed(reserved))
                    return reserved;
            // The target is sealed, so the shard generation opens closed like the rest, past the
            // seal.
            return target.shards_.open_shard_generation_past_seal_(
                target.shards_.open_shard_generation(),
                [&target](ram_shard_t const& built) noexcept { return target.build_shard_(built); });
        };
        if (status_t const packed = target.shards_.pack(to_writable_span(refining.landings_), 0, capacity, source_shard,
                                                        placement_of_t {}, rank_first_fit, open);
            failed(packed))
            return {{}, packed};
        if (status_t const sized = ram_indexes_t::shards_t::size_parts(
                to_span(refining.landings_), target.shards_.size(), refining.part_offsets_, refining.part_bases_);
            failed(sized))
            return {{}, sized};
        std::size_t const shards = refining.part_offsets_.size() - 1;
        if (failed(refining.shard_plans_.reserve_exactly(shards)) || failed(refining.cursors_.resize(shards)))
            return {{}, status_t::bad_alloc_k};
        if (status_t const binned = plan.cut_bins_(shards); failed(binned)) return {{}, binned};
        if (status_t const opened = plan.open_folds_(device_executor, *shaped); failed(opened)) return {{}, opened};
        return {std::move(plan), status_t::success_k};
    }

    /** Cuts the landings into bins: every source's non-empty shards landing in one of the @p shards
     *  target shards, in landing order, each bin closed once its widest shard outgrows the row bits
     *  one more member would leave, or once one more would leave each other member no slot of a
     *  row; then sizes the host columns for the widest bin. */
    status_t cut_bins_(std::size_t shards) noexcept {
        std::size_t const count = refining_.landings_.size();
        if (failed(members_.reserve_exactly(count)) || failed(bins_.reserve_exactly(count)))
            return status_t::bad_alloc_k;
        std::size_t first = 0;
        for (vram_indexes_t const* source : sources_) {
            std::size_t const source_shards = source->shards_.size(), connectivity = source->connectivity();
            for (std::size_t into = 0; into != shards; ++into) {
                bin_t bin {members_.size(), 0};
                std::size_t widest = 0;
                for (std::size_t from = first; from != first + source_shards; ++from) {
                    std::size_t const rows = source->shards_[from - first].index->size();
                    if (refining_.landings_[from].into != into || !rows) continue;
                    if (bin.count && ((max_of(widest, rows) >> row_bits_(bin.count + 1)) || connectivity < bin.count)) {
                        bins_.push_back_reserved(bin);
                        bin = {members_.size(), 0}, widest = 0;
                    }
                    members_.push_back_reserved(from);
                    widest = max_of(widest, rows), ++bin.count;
                }
                if (bin.count) bins_.push_back_reserved(bin);
            }
            first += source_shards;
        }
        std::size_t staged_bytes = 0;
        for (bin_t const& bin : bins_)
            staged_bytes = max_of(staged_bytes,
                                  bin_rows_(bin) * staged_row_bytes_(*shard_at_(members_[bin.first]).index));
        return failed(columns_.resize_uninitialized(staged_bytes)) ? status_t::bad_alloc_k : status_t::success_k;
    }

    /**
     *  @brief Grants every folding bin's devices its members' storage, then builds each member's
     *      walk context on its device's stream and its buffer and tables, mapped for every device
     *      of the bin; a bin of one member folds nothing and holds none of them.
     *
     *  A member's buffer holds its copy of the bin's shard table and fold table, then its fold
     *  rows, its picks and its links, @b [rows,connectivity] each, then the heads of its chains
     *  onto every other member, one run per member in bin order.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k A device of a bin that cannot reach another member's storage.
     */
    status_t open_folds_(device_executor_t& device_executor, vram_indexes_t const& shaped) noexcept {
        std::size_t states = 0;
        for (bin_t& bin : bins_)
            if (bin.count > 1) bin.state = states, states += bin.count;
        if (failed(folds_.reserve_exactly(states)) || failed(bin_tables_.resize(states)) ||
            failed(fold_tables_.resize(states)) || failed(arenas_.reserve_exactly(states)) ||
            failed(walks_.reserve_exactly(states)))
            return status_t::bad_alloc_k;
        std::size_t const connectivity = shaped.connectivity();
        for (bin_t const& bin : bins_) {
            if (bin.count == 1) continue;
            std::bitset<gpu_devices_capacity_k> devices;
            for (std::size_t member = bin.first; member != bin.first + bin.count; ++member) {
                std::size_t const device = static_cast<std::size_t>(
                    shard_at_(members_[member]).index->device_specs().ordinal);
                if (device >= devices.size()) return status_t::config_mismatch_k;
                devices[device] = true;
            }
            std::size_t const rows = bin_rows_(bin);
            std::size_t const table_bytes = bin.count * (sizeof(vram_bin_shard_t) + sizeof(vram_fold_shard_t));
            search_config_t search;
            search.expansion_factor = refining_.config_.expansion_factor;
            search.count = cross_count_(bin.count, connectivity);
            std::size_t first = 0;
            for (std::size_t member = bin.first; member != bin.first + bin.count; ++member) {
                vram_index_t& index = *shard_at_(members_[member]).index;
                std::size_t const own_rows = index.size(), state = bin.state + member - bin.first;
                int const device = index.device_specs().ordinal;
                // Every device of the bin scores this member's vectors and norms.
                if (status_t const granted = index.grant_(devices); failed(granted)) return granted;
                if (failed(cuda_vendor_t::select(index.api(), device))) return status_t::device_error_k;
                // The member's searches by the bin's other rows, then the fold's slices, if shared
                // memory cannot hold them.
                std::size_t const walk_bytes =
                    index.traversal_memory_usage(min_of(vram_indexes_t::window_rows_k, rows), search) +
                    std::size_t {index.device_specs().multiprocessor_count} *
                        vram_index_t::kernels_t::resident_blocks_(vram_dense_kernel_t::fold_k, 0) *
                        vram_warps_per_block_k *
                        gpu_arena_shape(0, 0, index.build_scratch_bytes_per_warp(index.core().reselection_nodes_()))
                            .template slice_bytes<node_id_t>() +
                    default_alignment_k;
                typename device_executor_t::stream_t& stream = stream_at_(device_executor, members_[member]);
                arenas_.push_back_reserved(typename device_executor_t::arena_t {stream.arena.upstream()});
                if (failed(arenas_.back().reserve_exactly(walk_bytes))) return status_t::bad_alloc_k;
                walks_.push_back_reserved(
                    walk_context_t {typename walk_context_t::allocator_t {arenas_.back(), stream}});
                if (!walks_.back()) return status_t::bad_alloc_k;

                std::size_t const row_words = own_rows * connectivity;
                std::size_t const bytes = table_bytes + (3 * row_words + rows - own_rows) * sizeof(u32_t);
                folds_.push_back_reserved(cuda_virtual_memory_allocator<byte_t> {index.api(), device});
                cuda_virtual_memory_allocator<byte_t>& buffer = folds_.back();
                if (status_t const granted = buffer.grant(devices); failed(granted)) return granted;
                if (failed(buffer.reserve_range(bytes)) || failed(buffer.commit(bytes))) return status_t::bad_alloc_k;
                vram_flat_storage_t const storage = index.core().storage();
                u32_t* const neighbors = reinterpret_cast<u32_t*>(buffer.base() + table_bytes);
                bin_tables_[state] = {storage.vectors, storage.norms, neighbors, static_cast<u32_t>(first),
                                      static_cast<u32_t>(own_rows)};
                fold_tables_[state] = {storage.neighbors, neighbors + row_words, neighbors + 2 * row_words,
                                       neighbors + 3 * row_words};
                first += own_rows;
            }
        }
        return status_t::success_k;
    }

    /** The fold launch of the shard at @p place in @p bin for @p phase, over its copies of the
     *  bin's tables. */
    vram_fold_t fold_launch_(bin_t const& bin, std::size_t place, vram_fold_phase_t phase) const noexcept {
        vram_index_t const& index = *shard_at_(members_[bin.first + place]).index;
        vram_bin_shard_t const* const bin_shards = reinterpret_cast<vram_bin_shard_t const*>(
            folds_[bin.state + place].base());
        vram_fold_t fold;
        fold.storage = index.core().storage();
        fold.bin_shards = bin_shards;
        fold.fold_shards = reinterpret_cast<vram_fold_shard_t const*>(bin_shards + bin.count);
        fold.shards_count = static_cast<u32_t>(bin.count);
        fold.shard = static_cast<u32_t>(place);
        fold.row_bits = row_bits_(bin.count);
        fold.cross_count = cross_count_(bin.count, index.connectivity());
        fold.selection_nodes = index.core().reselection_nodes_();
        fold.phase = phase;
        return fold;
    }

    /** Queues the current phase on every folding bin's members: the tables, the empty heads and
     *  every member's searches of the others for the cross phase, one fold launch per member for
     *  the others. Stops at the first refusal. */
    status_t queue_phase_() noexcept {
        for (bin_t const& bin : bins_) {
            if (bin.count == 1) continue;
            search_config_t search;
            search.expansion_factor = refining_.config_.expansion_factor;
            search.count = cross_count_(bin.count, shard_at_(members_[bin.first]).index->connectivity());
            for (std::size_t place = 0; place != bin.count; ++place) {
                vram_index_t& index = *shard_at_(members_[bin.first + place]).index;
                std::size_t const connectivity = index.connectivity();
                walk_context_t& walk = walks_[bin.state + place];
                if (failed(cuda_vendor_t::select(walk.api(), walk.device()))) return status_t::device_error_k;
                if (phase_ != phase_t::cross_k) {
                    vram_fold_phase_t const phase = phase_ == phase_t::reselect_k ? vram_fold_phase_t::reselect_k
                                                                                  : vram_fold_phase_t::merge_k;
                    if (status_t const launched = index.core().kernels().launch_fold_(
                            index.core(), fold_launch_(bin, place, phase), walk);
                        failed(launched))
                        return launched;
                    continue;
                }
                vram_bin_shard_t* const bin_shards = reinterpret_cast<vram_bin_shard_t*>(
                    folds_[bin.state + place].base());
                vram_fold_shard_t* const fold_shards = reinterpret_cast<vram_fold_shard_t*>(bin_shards + bin.count);
                vram_fold_shard_t const& own = fold_tables_[bin.state + place];
                std::size_t const heads = bin_rows_(bin) - index.size();
                if (failed(cuda_vendor_t::copy(walk.api(), bin_shards, &bin_tables_[bin.state],
                                               bin.count * sizeof(vram_bin_shard_t), walk.stream())) ||
                    failed(cuda_vendor_t::copy(walk.api(), fold_shards, &fold_tables_[bin.state],
                                               bin.count * sizeof(vram_fold_shard_t), walk.stream())) ||
                    failed(cuda_vendor_t::fill(walk.api(), own.heads, 0xFF, heads * sizeof(u32_t), walk.stream())))
                    return status_t::device_error_k;

                // This member's rows search every other member on its device, landing in its picks.
                vram_flat_storage_t const storage = index.core().storage();
                std::size_t const rows = index.size(), row_bytes = index.row_bytes();
                for (std::size_t other = 0; other != bin.count; ++other) {
                    if (other == place) continue;
                    vram_index_t& searched = *shard_at_(members_[bin.first + other]).index;
                    walk_context_t& searching = walks_[bin.state + other];
                    std::size_t const block = other - (other > place);
                    if (failed(cuda_vendor_t::select(searching.api(), searching.device())))
                        return status_t::device_error_k;
                    for (std::size_t first = 0; first < rows; first += vram_indexes_t::window_rows_k) {
                        std::size_t const window = min_of(vram_indexes_t::window_rows_k, rows - first);
                        rows_view_t const queries {storage.vectors + first * row_bytes, window, index.dimensions(),
                                                   row_bytes, vram_indexes_t::vector_dtype_k};
                        nk::matrix_span<node_id_t> const found {
                            reinterpret_cast<char*>(own.picks + first * connectivity + block * search.count),
                            rows_shape_of<node_id_t>(window, connectivity * sizeof(node_id_t), search.count)};
                        if (status_t const searched_status =
                                searched.search(queries, found, {}, searching, search, completion_t::deferred_k).status;
                            failed(searched_status))
                            return searched_status;
                    }
                }
            }
        }
        return status_t::success_k;
    }

    /** Drives the fold's pending phases, each queued on every folding bin and awaited on every
     *  member's context before the next, so a phase reads only what the last one landed; the
     *  tallies of the re-selection and the merge join the report. A phase failing partway leaves
     *  some members past it, so the fold restarts at the cross phase, which refills every head and
     *  pick. */
    template <typename progress_type_>
    status_t fold_(progress_type_& on_visited) noexcept {
        while (phase_ != phase_t::land_k) {
            status_t const queued = queue_phase_();
            // Every launch queued so far lands before anything returns, as the buffers die with
            // the plan.
            status_t waited = status_t::success_k;
            for (walk_context_t& walk : walks_)
                if (status_t const drained = walk.wait(); failed(drained) && succeeded(waited)) waited = drained;
            if (failed(queued) || failed(waited)) {
                phase_ = phase_t::cross_k;
                return failed(queued) ? queued : waited;
            }
            // The cross phase only searches, so only the re-selection and the merge count.
            for (bin_t const& bin : bins_) {
                if (phase_ == phase_t::cross_k || bin.count == 1) continue;
                for (std::size_t place = 0; place != bin.count; ++place) {
                    vram_telemetry_block_t const spent = walks_[bin.state + place].spent();
                    folded_.telemetry.computed_distances += spent.scored;
                    folded_.changed_nodes += spent.changed_nodes;
                    if (phase_ == phase_t::reselect_k)
                        folded_.reselected_nodes += shard_at_(members_[bin.first + place]).index->size();
                }
            }
            phase_ = static_cast<phase_t>(static_cast<u8_t>(phase_) + 1);
            if (phase_ != phase_t::reselect_k &&
                on_visited(folded_.reselected_nodes, moved_entries_) == continuation_t::cancel_k)
                return status_t::cancelled_k;
        }
        return status_t::success_k;
    }

    /** Lands every pending source into @p target, each source's landings the run of them its
     *  shards number. */
    status_t land_pending_(ram_indexes_t& target, device_executor_t& device_executor,
                           host_executor_t& host_executor) noexcept {
        std::size_t first = 0;
        for (std::size_t source = 0; source != landed_; ++source) first += sources_[source]->shards_.size();
        for (; landed_ != sources_.size(); ++landed_) {
            vram_indexes_t& source = *sources_[landed_];
            std::size_t const count = source.shards_.size();
            if (status_t const landed = land_(target, source, first, device_executor, host_executor); failed(landed))
                return landed;
            first += count;
        }
        return status_t::success_k;
    }

    /**
     *  @brief Lands every row of @p source, whose shards number from @p first across the sources,
     *      where the landings put it, and consumes it.
     *
     *  The source's gate and every other fallible step come first, then the copies, then one commit
     *  bracket publishes and the source shards join the plan's parts. Every target shard claims all
     *  the rows it takes up front, so a failure before publication releases them. Each bin then
     *  crosses to the host columns side by side: every member's save plan stages its vectors and
     *  neighbors on its device's stream, a folding bin's fold rows replace the neighbors, already
     *  in the bin's dense ID space, and the target shard's load steps covering the bin copy out of
     *  the columns on the host threads, the keys beside them. Once every bin landed, the target
     *  publishes, the keys resolve, and the source forgets its rows in place inside its window.
     */
    status_t land_(ram_indexes_t& target, vram_indexes_t& source, std::size_t first, device_executor_t& device_executor,
                   host_executor_t& host_executor) noexcept {
        std::size_t const shards = refining_.cursors_.size();
        if (!source.size()) return status_t::success_k;
        std::span<landing_t const> const landings = to_span(refining_.landings_).subspan(first, source.shards_.size());
        // The source empties once its rows land, so its gate precedes anything a refusal undoes.
        typename vram_indexes_t::unique_lock_t const emptying {source.gate_, std::try_to_lock};
        if (!emptying) return status_t::contended_k;
        if (!target.capacity())
            if (status_t const reserved = target
                                              .reserve({source.size(),
                                                        static_cast<connectivity_t>(source.connectivity()),
                                                        source.dimensions()})
                                              .status;
                failed(reserved))
                return reserved;

        std::size_t const sources = landings.size();
        growing_buffer<ram_load_plan_t, forkunion_allocator_t> plans {target.shards_.allocator()};
        growing_buffer<std::size_t, forkunion_allocator_t> cursors {target.shards_.allocator()};
        if (failed(plans.reserve_exactly(shards)) || failed(cursors.resize(shards))) return status_t::bad_alloc_k;
        std::size_t landing_rows = 0;
        for (std::size_t into = 0; into != shards; ++into) {
            std::size_t rows = 0;
            for (std::size_t from = 0; from != sources; ++from)
                if (landings[from].into == into) rows += source.shards_[from].index->size();
            auto [plan, status] = target.shards_[into].index->plan_load(rows);
            if (failed(status)) return status;
            plans.push_back_reserved(std::move(plan));
            if (status_t const covered = target.shards_[into].keys.cover(plans[into].base + rows); failed(covered))
                return covered;
            landing_rows += rows;
        }
        auto [seats, seat_status] = target.shards_.locators().claim(landing_rows);
        if (failed(seat_status)) return seat_status;

        std::size_t const connectivity = source.connectivity();
        std::size_t const vectors_row_bytes = packed_bytes(vram_indexes_t::vector_dtype_k, source.dimensions());
        std::size_t const neighbors_row_bytes = connectivity * sizeof(node_id_t);
        for (bin_t const& bin : bins_) {
            if (members_[bin.first] < first || members_[bin.first] >= first + sources) continue;
            std::size_t const into = landings[members_[bin.first] - first].into;
            node_id_t const rows = static_cast<node_id_t>(bin_rows_(bin));
            node_id_t landed = plans[into].base;
            for (std::size_t earlier = 0; earlier != members_[bin.first] - first; ++earlier)
                if (landings[earlier].into == into)
                    landed += static_cast<node_id_t>(source.shards_[earlier].index->size());
            iota_view<node_id_t> const landing {landed, static_cast<node_id_t>(landed + rows)};
            byte_t* const vectors = columns_.data();
            byte_t* const neighbors = vectors + rows * vectors_row_bytes;

            node_id_t offset = 0;
            for (std::size_t member = bin.first; member != bin.first + bin.count; ++member) {
                std::size_t const from = members_[member] - first;
                vram_shard_t const& part = source.shards_[from];
                node_id_t const part_rows = static_cast<node_id_t>(part.index->size());
                typename vram_index_t::columns_t const staged {
                    bytes_of(vectors + offset * vectors_row_bytes, part_rows * vectors_row_bytes),
                    bytes_of(neighbors + offset * neighbors_row_bytes, part_rows * neighbors_row_bytes),
                    {}};
                typename device_executor_t::stream_t& stream = device_executor.stream(
                    fanout_t::stream_of(source.shards_, device_executor, from));
                if (status_t const entered = fanout_t::enter(source.shards_, from); failed(entered)) return entered;
                auto [saving, save_status] = part.index->plan_save(staged, synchronization_t::immutable_k, stream);
                if (failed(save_status)) return save_status;
                for (typename vram_index_t::save_plan_t::step_t const step : saving.steps())
                    if (status_t const saved = saving.save(step, {}); failed(saved)) return saved;
                if (status_t const saved = saving.finish(); failed(saved)) return saved;
                // A folding bin's own rows replace the live ones the save staged.
                std::size_t const state = bin.state + member - bin.first;
                if (bin.count > 1 && (failed(cuda_vendor_t::copy(walks_[state].api(), staged.neighbors.data(),
                                                                 bin_tables_[state].neighbors, staged.neighbors.size(),
                                                                 walks_[state].stream())) ||
                                      failed(walks_[state].wait())))
                    return status_t::device_error_k;
                offset += part_rows;
            }

            // A target step is one storage chunk, which may straddle this bin and its neighbors.
            typename ram_index_t::columns_view_t const view {
                bytes_of(static_cast<byte_t const*>(vectors), rows * vectors_row_bytes),
                bytes_of(static_cast<byte_t const*>(neighbors), rows * neighbors_row_bytes),
                {}};
            auto land = [&](std::size_t shard, typename ram_load_plan_t::step_t const& chunk, std::size_t,
                            nk::matrix_span<node_id_t>) noexcept -> status_t {
                if (shard != into) return status_t::success_k;
                typename ram_load_plan_t::step_t step = chunk;
                step.index_nodes = {max_of(chunk.index_nodes.first, landing.first),
                                    min_of(chunk.index_nodes.bound, landing.bound)};
                if (step.index_nodes.first >= step.index_nodes.bound) return status_t::success_k;
                step.source_nodes = {static_cast<node_id_t>(step.index_nodes.first - landing.first),
                                     static_cast<node_id_t>(step.index_nodes.bound - landing.first)};
                if (status_t const loaded = plans[into].load(step, view, {0, rows}); failed(loaded)) return loaded;
                auto part_of = [&](std::size_t member) noexcept -> vram_shard_t const& {
                    return source.shards_[members_[member] - first];
                };
                // The rows ascend, so the member holding each only moves forward.
                std::size_t member = bin.first;
                node_id_t member_first = landing.first;
                for (node_id_t row : step.index_nodes) {
                    while (row - member_first >= part_of(member).index->size())
                        member_first += static_cast<node_id_t>(part_of(member++).index->size());
                    target.shards_[into].keys[row] = part_of(member).keys.host[row - member_first];
                }
                return status_t::success_k;
            };
            std::memset(cursors.data(), 0, shards * sizeof(std::size_t));
            if (status_t const loaded = target.drain_steps_(plans, to_writable_span(cursors), host_executor, 0, land);
                failed(loaded))
                return loaded;
        }

        // Every bin landed, so no plan refuses; the bracket shows a search every shard's rows or
        // none, and the keys resolve only once their rows are visible.
        auto publish = [&]() noexcept {
            for (ram_load_plan_t& plan : plans)
                if (status_t const finished = plan.finish(); failed(finished)) return finished;
            target.shards_.record_parts(source.shards_, landings, plans, refining_.part_bases_.data(),
                                        to_span(refining_.part_offsets_));
            return status_t::success_k;
        };
        target.shards_.begin_commit();
        status_t const published = publish();
        target.shards_.end_commit();
        if (failed(published)) return published;
        for (std::size_t into = 0; into != shards; ++into)
            for (node_id_t row = plans[into].base; row != plans[into].base + plans[into].rows; ++row)
                target.shards_.locators().insert(seats, target.shards_[into].keys[row],
                                                 typename ram_indexes_t::key_locator_t {static_cast<u32_t>(into), row});
        source.shards_.begin_window();
        source.clear_();
        source.shards_.end_window();
        return status_t::success_k;
    }

    /** Reopens every source still pending, each closed at planning; the plan holds none after. */
    void release_sources_() noexcept {
        for (std::size_t pending = landed_; pending < sources_.size(); ++pending) sources_[pending]->shards_.reopen();
        sources_ = {};
    }

    /** Runs and finishes the plan on host contexts carved for the call: the one-shot
     *  @ref compact. */
    template <typename progress_type_>
    expected<result_t> drive_(device_executor_t& device_executor, host_executor_t& host_executor,
                              progress_type_& on_visited) noexcept {
        typename ram_indexes_t::carved_traversals_t traversals {
            arena_allocator<traversal_context_t, forkunion_allocator_t>(host_executor.arena(0))};
        if (status_t const carved = refining_.owner_->carve_traversals_(traversals, host_executor); failed(carved))
            return {{}, carved};
        status_t const ran = run(device_executor, host_executor, to_writable_span(traversals), on_visited);
        auto const [result, finish_status] = finish(host_executor, to_writable_span(traversals));
        return {result, failed(finish_status) ? finish_status : ran};
    }

    /** The absorb plan over the target: its landings, part runs and per-shard heals. */
    refine_plan_t refining_;

    /** One bin's packed vectors, then its neighbors, sized at planning for the widest, which every
     *  bin crosses to the host through. */
    growing_buffer<byte_t, forkunion_allocator_t> columns_ {forkunion_allocator_t {}};

    /** Every non-empty source shard, numbered across the sources, in bin order. */
    growing_buffer<std::size_t, forkunion_allocator_t> members_ {forkunion_allocator_t {}};

    /** The bins, each a run of @c members_; one of a single member folds nothing and holds no
     *  fold state. */
    growing_buffer<bin_t, forkunion_allocator_t> bins_ {forkunion_allocator_t {}};

    /** Every folding member's fold buffer on its device, mapped for every device of its bin, from
     *  its bin's state on; the tables, runs and contexts below are laid out alike. */
    growing_buffer<cuda_virtual_memory_allocator<byte_t>, forkunion_allocator_t> folds_ {forkunion_allocator_t {}};

    /** Every folding member's entries of its bin's shard table and fold table, as the cross phase
     *  uploads them into each member's buffer. */
    growing_buffer<vram_bin_shard_t, forkunion_allocator_t> bin_tables_ {forkunion_allocator_t {}};
    growing_buffer<vram_fold_shard_t, forkunion_allocator_t> fold_tables_ {forkunion_allocator_t {}};

    /** Every folding member's run of device scratch on its stream, and the context its searches and
     *  fold launches ride; the runs die after the contexts carved from them. */
    growing_buffer<typename device_executor_t::arena_t, forkunion_allocator_t> arenas_ {forkunion_allocator_t {}};
    growing_buffer<walk_context_t, forkunion_allocator_t> walks_ {forkunion_allocator_t {}};

    /** The caller's source slots, borrowed until the plan dies. */
    std::span<vram_indexes_t* const> sources_;

    /** Sources landed so far. */
    std::size_t landed_ = 0;

    /** Source rows, every one of which changes shard. */
    std::size_t moved_entries_ = 0;

    /** The fold's re-selections and merges, as a refine counts them. */
    refine_result_t folded_;

    /** The fold's next phase. */
    phase_t phase_ = phase_t::cross_k;
};

template <typename vram_indexes_type_, typename ram_indexes_type_>
expected<vram_nvidia_compact_plan<vram_indexes_type_, ram_indexes_type_>> plan_compact(
    std::span<vram_indexes_type_* const> sources, ram_indexes_type_& target, cuda_executor_t& device_executor,
    typename ram_indexes_type_::executor_t& host_executor, compact_config_t const& config) noexcept {
    return vram_nvidia_compact_plan<vram_indexes_type_, ram_indexes_type_>::open_(sources, target, device_executor,
                                                                                  host_executor, config);
}

template <typename vram_indexes_type_, typename ram_indexes_type_, typename progress_type_>
expected<compact_result_t> compact(std::span<vram_indexes_type_* const> sources, ram_indexes_type_& target,
                                   cuda_executor_t& device_executor,
                                   typename ram_indexes_type_::executor_t& host_executor,
                                   compact_config_t const& config, progress_type_&& on_visited) noexcept {
    auto [plan, status] = plan_compact(sources, target, device_executor, host_executor, config);
    if (failed(status)) return {{}, status};
    return plan.drive_(device_executor, host_executor, on_visited);
}

} // namespace unum::usearch::dense
