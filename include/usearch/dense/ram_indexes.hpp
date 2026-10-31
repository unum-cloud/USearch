/**
 *  @file include/usearch/dense/ram_indexes.hpp
 *  @author Ash Vardanian
 *  @date July 23, 2026
 *  @brief The sharded index over CPU-resident dense shards - one per memory domain - with
 *      rotation-balanced insertion, merged cross-shard search, and SafeGraphs checkpoints.
 *
 *  The sharded layer over @c ram_index: same-geometry shards, each owned and pinned to one
 *  ForkUnion memory domain, viewed as one index. Insertions rotate across the shards so they fill
 *  evenly; searches scatter over every shard and gather through
 *  @ref unum::usearch::utilities::merge_search_results, resolving matches into the caller's keys,
 *  so shards with colliding ID spaces merge into one ranking. Checkpoints ride @c shard_save_plan
 *  and @c shard_load_plan, every step fanned out over the executor's threads.
 */
#pragma once

#include <cassert> // `assert`
#include <cstring> // `std::memset`, `std::memcpy`, `std::memmove`, `std::memcmp`

#include <array>        // `std::array`
#include <limits>       // `std::numeric_limits` - the open threshold of a short exact ranking
#include <memory>       // `std::unique_ptr`
#include <numbers>      // `std::numbers::log2e_v` - a scanned logit in base 2
#include <mutex>        // `std::lock_guard` - the dispensing mutex, `std::unique_lock` - the windows' hold
#include <shared_mutex> // `std::shared_lock` - the mutators' hold on the gate
#include <span>         // `std::span`
#include <string_view>  // `std::string_view`
#include <type_traits>  // `std::is_same_v`, `std::invoke_result_t`
#include <utility>      // `std::move`, `std::swap`

#include <forkunion/forkunion.h> // `fu_memory_domain_id_t`
#include <forkunion/atomics.hpp> // `fu::standard_atomic_ref` - the default reference

#include "usearch/utilities/types.hpp"
#include "usearch/utilities/ram_buffers.hpp"       // `growing_buffer` - the plans' per-thread state
#include "usearch/utilities/ram_device.hpp"        // `forkunion_allocator`, `forkunion_executor_t`, `shared_mutex`
#include "usearch/utilities/safegraphs_format.hpp" // `save_config_t`, `load_config_t`, `safegraphs_read_metadata`
#include "usearch/utilities/ram_shards.hpp"        // `shard_set` - the shards & their routing
#include "usearch/utilities/ram_walks.hpp"         // `breadth_first_walk`, `cursor_order_t` - the cursor's walk
#include "usearch/utilities/sample.hpp"            // `sample_plan` - the SAMPLE verb's roots, paths & regions
#include "usearch/dense/types.hpp"
#include "usearch/dense/shard_checkpoints.hpp"
#include "usearch/dense/ram_index.hpp"

namespace unum::usearch::dense {

template <typename indexes_type_>
struct ram_indexes_cursor;

template <typename indexes_type_>
struct ram_compact_plan;

template <typename vram_indexes_type_, typename ram_indexes_type_>
struct vram_nvidia_compact_plan;

template <typename vram_indexes_type_, typename ram_indexes_type_>
struct vram_amd_compact_plan;

/** Resolves a walked node ID to its key through one shard's keys, rebound per shard. */
template <typename keys_type_>
struct keyed_ids {

    /** The keys of the shard being walked. */
    keys_type_ const* keys = nullptr;

    auto operator()(node_id_t node_id) const noexcept { return (*keys)[node_id]; }
};

/**
 *  @brief The sharded index: same-geometry CPU-resident shards viewed as one.
 *
 *  It owns its shards outright, so a destructor, a consuming merge and @ref clear are
 *  expressible. Shards arrive one at a time through @ref emplace, so the index grows as domains
 *  are engaged rather than being declared whole. The shard's configs, views and results are
 *  re-exported verbatim, so a batched call and a single one read the same.
 *
 *  Every verb is safe alongside every other. Mutators hold one gate's shared side for each call,
 *  and the destructive windows - a clear, each close, and an absorb, split or compaction emptying
 *  shards in place - take its unique side or answer @c contended_k. Readers take no lock: they
 *  bracket each call, or each row of a batch, with the shard set's epoch, retry once when a window
 *  overlapped them, and answer @c contended_k when a second one did. Nothing a reader can reach is
 *  freed while the index lives: emptied shards keep their memory until it dies.
 */
template <                                                                    //
    typename key_type_ = uuid_t,                                              //
    typename quads_kit_type_ = bf16_serial_quads_kit_t,                       //
    typename tiles_kit_type_ = no_tiles_kit_t,                                //
    typename elementwise_kit_type_ = serial_elementwise_kit_t,                //
    typename rmscast_kit_type_ = no_rmscast_kit_t,                            //
    typename visits_kit_type_ = serial_visits_hash_set_kit_t,                 //
    typename wait_type_ = fu::standard_yield_t,                               //
    template <typename> class atomic_ref_template_ = fu::standard_atomic_ref, //
    typename cache_hints_type_ = dummy_cache_hints_t,                         //
    typename key_hash_type_ = std::hash<key_type_>                            //
    >
struct ram_indexes {
    template <typename>
    friend struct ram_compact_plan;
    template <typename, typename>
    friend struct vram_nvidia_compact_plan;
    template <typename, typename>
    friend struct vram_amd_compact_plan;

    /** A reader's walk over every shard's published entries: as stored in placement order, or
     *  breadth-first from the entries it opens on, each with the keys it links to. */
    using cursor_t = ram_indexes_cursor<ram_indexes>;

    /** The shard type - geometry parameters in the shard's own order, so the two can never
     *  disagree. */
    using index_t =
        ram_index<quads_kit_type_, tiles_kit_type_, elementwise_kit_type_, rmscast_kit_type_, visits_kit_type_,
                  wait_type_, atomic_ref_template_, cache_hints_type_, forkunion_allocator_t, forkunion_allocator_t,
                  arena_allocator<byte_t, forkunion_allocator_t>, key_type_, key_hash_type_>;

    /** The caller-visible identity; every shard keeps its own nodes' keys and files their nodes. */
    using key_t = key_type_;

    /** One shard's keys and key map, which every walk resolves its nodes through. */
    using keyed_t = typename index_t::keyed_t;

    /** The dimensions and connectivity every shard keeps, fixed at construction. */
    using geometry_t = typename index_t::geometry_t;

    /** The shards, the rotation that routes new rows, and the commit word a reader spanning the
     *  shards snapshots by. */
    using shards_t = shard_set<index_t, atomic_ref_template_, wait_type_>;
    using shard_t = typename shards_t::shard_t;
    using shard_generation_t = typename shards_t::shard_generation_t;
    using batch_split_t = typename shards_t::batch_split_t;

    template <typename word_type_>
    using atomic_ref = atomic_ref_template_<word_type_>;

    /** The threads and their scratch every bulk verb runs on; it names no index, so one executor
     *  serves every instantiation. */
    using executor_t = forkunion_executor_t;
    using traversal_context_t = typename index_t::traversal_context_t;

    /* Scalars and views */
    using distance_t = typename index_t::distance_t;
    using scalar_t = typename index_t::scalar_t;
    using dot_result_t = typename scalar_t::dot_result_t;
    using vector_view_t = typename index_t::vector_view_t;
    using vectors_view_t = typename index_t::vectors_view_t;
    using vector_norms_t = typename index_t::vector_norms_t;

    /* Verb configs */
    using insert_config_t = typename index_t::insert_config_t;
    using search_config_t = typename index_t::search_config_t;
    using refine_config_t = typename index_t::refine_config_t;
    using split_config_t = typename index_t::split_config_t;

    /** A MaxSim's knobs: its walks', and the keys each query keeps, zero keeping every key it
     *  scored. */
    struct maxsim_config_t {
        search_config_t search {};
        std::size_t wanted = 0;
    };
    using reserve_capacity_t = typename index_t::reserve_capacity_t;

    /* Verb results */
    using insert_result_t = typename index_t::insert_result_t;
    using search_result_t = typename index_t::search_result_t;
    using refine_report_t = typename index_t::refine_report_t;
    using reachability_t = typename index_t::reachability_t;
    using shard_capacity_claim_t = typename index_t::capacity_claim_t;

    /* Predicate payloads */
    using walk_payload_t = typename index_t::walk_payload_t;
    using scan_payload_t = dense::scan_payload_t;
    static_assert(index_family<index_t> && index_claim<shard_capacity_claim_t>);

    /** The stored scalar vocabulary, re-exported for dispatchers gating crossing rows. */
    inline static constexpr nk_dtype_t vector_dtype_k = index_t::vector_dtype_k;

    /** The kits every shard scores and imports through, as @c src/ picked them. */
    using quads_kit_t = quads_kit_type_;
    using tiles_kit_t = tiles_kit_type_;
    using elementwise_kit_t = elementwise_kit_type_;

#pragma region Placement

    /** Every index names the distance kind its shards walk by, the geometry they keep, the host
     *  allocator its shard table and checkpoint catalogs sit on, and the line its shards prefetch
     *  by; nothing is placed by default. */
    explicit ram_indexes(nk_kernel_kind_t kernel, geometry_t geometry, forkunion_allocator_t allocator,
                         std::size_t line_bytes = default_alignment_k) noexcept
        : shards_(allocator), kernel_(kernel), geometry_(geometry), line_bytes_(line_bytes) {}
    ram_indexes(ram_indexes&& other) noexcept
        : shards_(std::move(other.shards_)), kernel_(other.kernel_), geometry_(other.geometry_),
          line_bytes_(other.line_bytes_) {}
    ram_indexes& operator=(ram_indexes&& other) noexcept { return swap(other), *this; }
    ram_indexes(ram_indexes const&) = delete;
    ram_indexes& operator=(ram_indexes const&) = delete;

    /** Every member is RAII, so the shards die with the index and nothing leaks. */
    ~ram_indexes() noexcept = default;

    /** Exchanges two indexes whole; a move implies the caller holds both exclusively. */
    void swap(ram_indexes& other) noexcept {
        shards_.swap(other.shards_);
        std::swap(kernel_, other.kernel_);
        std::swap(geometry_, other.geometry_);
        std::swap(line_bytes_, other.line_bytes_);
    }

    /** Bounds every shard at @p rows, clamped to the ID space; zero means the ID space. Past it, an
     *  insertion opens another shard generation on the same placements. Ignored once storage
     *  exists. */
    void bound_shards(std::size_t rows) noexcept { shards_.bound_shards(rows); }

    /** Rows one shard may hold. */
    std::size_t shard_capacity() const noexcept { return shards_.shard_capacity(); }

    /**
     *  Appends one shard built here on @p memory_domain in the index's geometry, for callers with
     *  none to hand over.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k The set is reserved, retired shards, or has a shard generation in
     *    flight.
     */
    status_t emplace(fu_memory_domain_id_t memory_domain) noexcept {
        std::unique_ptr<index_t> shard = build_(memory_domain);
        if (!shard) return status_t::bad_alloc_k;
        return emplace(std::move(shard));
    }

    /** Appends one shard the caller already built, consuming it; refuses any walking by another
     *  kernel or keeping another geometry, and any already reserved. Refused once storage exists -
     *  @ref reserve closes the shard set.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k @p shard is null, walks by another kernel, differs in geometry, or
     *    is already reserved.
     *  - @c config_mismatch_k The set is reserved, retired shards, or has a shard generation in
     *    flight.
     */
    status_t emplace(std::unique_ptr<index_t>&& shard) noexcept {
        if (!shard || shard->kernel() != kernel_ || shard->geometry() != geometry_) return status_t::config_mismatch_k;
        return shards_.emplace(std::move(shard));
    }

    /**
     *  @brief Empties every shard in place, keeping its memory domain, geometry and capacity, and
     *      every shard generation past the first as idle shards the next ones reuse.
     *
     *  Takes the gate alone and closes every shard first, so a call in flight, a batch holding
     *  claims or a plan holding the index closed refuses it with @c contended_k, and a search it
     *  overlaps retries around it.
     *
     *  @par Statuses:
     *  - @c contended_k A call holds the gate, or a batch or plan holds a shard's dispenser.
     */
    status_t clear() noexcept {
        unique_lock_t const held {gate_, std::try_to_lock};
        if (!held) return status_t::contended_k;
        if (status_t const closed = shards_.close(); failed(closed)) return closed;
        shards_.begin_window();
        clear_();
        shards_.end_window();
        shards_.reopen();
        return status_t::success_k;
    }

#pragma endregion Placement

#pragma region Measures and Reservation

    /** The distance kind every shard walks by, fixed at construction. */
    nk_kernel_kind_t kernel() const noexcept { return kernel_; }

    /** Shards in the index, one per engaged memory domain. */
    std::size_t shards_count() const noexcept { return shards_.size(); }

    /** The shard at @p shard, in placement order. */
    index_t& shard(std::size_t shard) noexcept { return *shards_[shard].index; }

    /** The read-only shard at @p shard. */
    index_t const& shard(std::size_t shard) const noexcept { return *shards_[shard].index; }

    /** The geometry every shard keeps - @ref emplace refused any other. */
    geometry_t geometry() const noexcept { return geometry_; }

    /** Scalar dimensions per row. */
    std::size_t dimensions() const noexcept { return geometry_.dimensions; }

    /** Connections per node. */
    std::size_t connectivity() const noexcept { return geometry_.connectivity; }

    /** Entries across every shard: the watermarks of one consistent snapshot, summed, so a batch
     *  counts whole or not at all. */
    std::size_t size() const noexcept { return shards_.entries(); }

    /** The epoch a reader starts under, waiting out a destructive window in progress. */
    u64_t read_epoch() const noexcept { return shards_.read_epoch(); }

    /** Whether no destructive window began since @p epoch, so everything read under it stands. */
    bool still_valid(u64_t epoch) const noexcept { return shards_.still_valid(epoch); }

    /** Rows the shards host without growing again; they grow past it on demand. */
    std::size_t capacity() const noexcept { return shards_.capacity(); }

    /** Bytes the index occupies - every shard's storage, its keys and its key map; traversal
     *  scratch is the executor's. */
    std::size_t memory_usage() const noexcept { return sizeof(ram_indexes) + shards_.memory_usage(); }

    /**
     *  @brief Grows every shard ahead of @p config.capacity rows, split evenly over them - a hint,
     *      since the shards and their keys grow on demand.
     *
     *  Callable any time, alongside insertions and searches: nothing it grows ever moves.
     *
     *  @par Statuses:
     *  - @c uninitialized_k No shard is placed yet.
     *  - @c unknown_kind_k The dtype's kits cannot finalize the kernel.
     *  - @c config_mismatch_k The geometry is malformed, or a nonzero one in @p config differs.
     *  - @c overflow_k A shard's share passes its ceiling or its chunk directory.
     */
    expected<std::size_t> reserve(reserve_capacity_t config) noexcept { return shards_.reserve(config); }

#pragma endregion Measures and Reservation

#pragma region Insertion

    /**
     *  @brief Inserts one row under @p key into whichever shard the rotation names next - the
     *      single-threaded sibling of the batched form, taking a context where it takes an executor.
     *
     *  Safe to call concurrently with itself, with the batched form, and with searches: the shard
     *  is chosen by one relaxed read-modify-write, and everything past the dispense is a step that
     *  cannot fail. It holds the gate's shared side for the call, and a refine, an absorb, a split,
     *  a compaction or a load holding the index closed answers @c contended_k.
     *
     *  @param[in] key The caller's identity for the row; several rows may share one.
     *  @param[in] new_vector The row, in the shards' stored scalar type.
     *  @param[in,out] traversal The calling thread's scratch & telemetry.
     *  @param[in] config The insertion configuration handed to the receiving shard.
     *
     *  @par Statuses:
     *  - @c uninitialized_k No shard is placed yet.
     *  - @c contended_k A plan holds the index closed, a sealed set must open a shard generation,
     *    or a shard's dispenser closed under the claim.
     *  - @c overflow_k A queue capacity of @p traversal wraps when rounded.
     */
    expected<insert_result_t> insert(key_t const& key, vector_view_t new_vector, traversal_context_t& traversal,
                                     insert_config_t config = {}) noexcept {
        insert_result_t result;
        result.new_size = size();
        if (shards_.empty()) return {result, status_t::uninitialized_k};
        shared_lock_t const held {gate_};
        if (shards_.sealed()) return {result, status_t::contended_k};

        // The routing tally only names the open shard generation's first shard to try; a full one
        // passes the row on to the next, a full shard generation opens the next one, and any other
        // refusal stops the probe.
        std::size_t const routing_base = shards_.route(1);
        shard_capacity_claim_t claim;
        std::size_t shard = 0;
        for (shard_generation_t open = shards_.open_shard_generation();; open = shards_.open_shard_generation()) {
            std::size_t probe = 0;
            for (; probe != open.width; ++probe) {
                shard = open.first + (routing_base + probe) % open.width;
                auto [shard_claim, claim_status] = shards_[shard].index->claim(1);
                if (succeeded(claim_status)) {
                    claim = std::move(shard_claim);
                    break;
                }
                if (claim_status != status_t::overflow_k) return {result, claim_status};
            }
            if (probe != open.width) break;
            if (status_t const opened = shards_.open_shard_generation_(
                    open, [this](shard_t const& like) noexcept { return build_shard_(like); });
                failed(opened))
                return {result, opened};
        }

        shard_t& slot = shards_[shard];
        if (status_t const sized = slot.index->reserve_traversal(traversal, config); failed(sized))
            return {result, sized};

        // The context's counters run across calls, so the report is the difference this one made.
        std::size_t const visited_before = traversal.visited_members;
        std::size_t const computed_before = traversal.computed_distances;
        std::size_t const refines_before = traversal.computed_distances_in_refines;
        std::size_t const reverse_refines_before = traversal.computed_distances_in_reverse_refines;
        std::size_t const restarts_before = traversal.restarts;

        // One shard's turn files the key with the row, so no bracket is needed.
        vectors_view_t const row {new_vector.data(), 1u, new_vector.size()};
        status_t const inserted = slot.index->insert(
            std::move(claim), typename index_t::keys_view_t {&key, std::size_t {1}}, row, {}, traversal, config);
        if (failed(inserted)) return {result, inserted};

        result.visited_members = traversal.visited_members - visited_before;
        result.computed_distances = traversal.computed_distances - computed_before;
        result.computed_distances_in_refines = traversal.computed_distances_in_refines - refines_before;
        result.computed_distances_in_reverse_refines = traversal.computed_distances_in_reverse_refines -
                                                       reverse_refines_before;
        result.restarts = traversal.restarts - restarts_before;
        result.new_size = size();
        return {result, status_t::success_k};
    }

    /**
     *  @brief Inserts a whole matrix of rows under @p keys, split across the shards from a rotating
     *      start so they fill evenly - the executor-driven sibling of one insertion.
     *
     *  Node IDs never cross this boundary: they are shard-local, and which shard a row went into
     *  is the index's business. The keys are what the caller gets its rows back by.
     *
     *  All-or-nothing across shards: every claim is taken before any row is placed, past that point
     *  nothing can fail, and a search sees the whole batch or none of it. One executor drives one
     *  call at a time.
     *
     *  @param[in] keys One caller identity per row; several rows may share one.
     *  @param[in] new_rows The batch, one row per entry, in any scalar NumKong casts into the
     *      stored one.
     *  @param[in,out] executor The threads and their scratch.
     *  @param[in] config The per-shard insertion configuration, applied uniformly.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k @p keys differs in length from the rows, or the rows' width from the
     *    dimensions.
     *  - @c uninitialized_k No shard is placed, or @p executor holds no arenas.
     *  - @c contended_k A plan holds the index closed, another call holds @p executor, a sealed set
     *    must open a shard generation, or a shard's dispenser closed under the claim.
     *  - @c overflow_k A racing batch took the room this one was routed into, or a queue capacity
     *    wraps when rounded.
     */
    expected<insert_result_t> insert(std::span<key_t const> keys, rows_view_t new_rows, executor_t& executor,
                                     insert_config_t config = {}) noexcept {
        insert_result_t result;
        result.new_size = size();
        if (keys.size() != new_rows.rows) return {result, status_t::config_mismatch_k};
        // Rows go into slots as wide as the geometry, and a row refused past the dispense would
        // stall its shard.
        if (new_rows.columns != dimensions()) return {result, status_t::config_mismatch_k};
        auto [plan, status] = plan_insert(new_rows.rows, executor, config);
        if (failed(status)) return {result, status};
        carved_traversals_t traversals {arena_allocator<traversal_context_t, forkunion_allocator_t>(executor.arena(0))};
        if (status_t const carved = carve_traversals_(traversals, executor); failed(carved)) return {result, carved};
        if (status_t const inserted = plan.insert(0, keys, new_rows, to_writable_span(traversals)); failed(inserted))
            return {result, inserted};
        return plan.finish();
    }

    /**
     *  @brief One batched insertion in flight: every fallible step taken, so its rows can only be
     *      placed, and @ref finish publishes them all at once.
     *
     *  Rows are split across the open shard generation's shards round-robin from the starting shard
     *  on, each taking the next ID of its shard's range; rows past what fits under the bound fill
     *  fresh shard generations in turn, split the same way. The plan holds only reversible claims
     *  until its first step dispenses the ranges, so a plan dropped before then releases
     *  everything, and a caller spanning several indexes plans on all of them before stepping any,
     *  committing or refusing as a whole. Once stepped, its ranges hold every later turn on their
     *  shards until it finishes, so it must finish. Planning, each step and @ref finish hold the
     *  gate's shared side for their own call alone, and claims outstanding refuse every close with
     *  @c contended_k.
     */
    struct [[nodiscard]] insert_plan_t {

        insert_plan_t() noexcept : insert_plan_t(forkunion_allocator_t {}) {}

        /** An empty plan over @p allocator; only @ref plan_insert ever fills it. */
        explicit insert_plan_t(forkunion_allocator_t allocator) noexcept : claims_(allocator), ranges_(allocator) {}
        insert_plan_t(insert_plan_t&& other) noexcept
            : claims_(std::move(other.claims_)), ranges_(std::move(other.ranges_)),
              executor_claim_(std::move(other.executor_claim_)), executor_(other.executor_), config_(other.config_),
              batch_split_(other.batch_split_), inserted_entries_(other.inserted_entries_), spent_(other.spent_),
              owner_(std::exchange(other.owner_, nullptr)) {}
        insert_plan_t& operator=(insert_plan_t&&) = delete;
        insert_plan_t(insert_plan_t const&) = delete;
        insert_plan_t& operator=(insert_plan_t const&) = delete;

        /** An unstepped plan gives back the executor and every shard's claim as its members die; a
         *  stepped one holds its shards' turns, so it must finish. */
        ~insert_plan_t() noexcept { assert((!owner_ || ranges_.empty()) && "A dispensed plan must finish"); }

        /**
         *  @brief Places rows @p first_row onward of the plan, one per row of @p rows under its key
         *      in @p keys, fanned out over the executor's threads, each walking its context of the
         *      driver's @p traversals and imported into the stored scalar as it is placed.
         *
         *  It fails only before its rows are dispensed.
         *
         *  The first step dispenses every shard's range under the dispensing mutex, so no two
         *  batches' ranges cross order between the shards. Rows outside the plan or of another
         *  width place nothing, and @ref finish refuses until every row is placed. One caller at a
         *  time, like the executor.
         *
         *  @par Statuses:
         *  - @c contended_k A refine, an absorb or a load holds a shard's dispenser closed;
         *    nothing was dispensed, so a later step may retry and a dropped plan releases its
         *    claims.
         *  - @c config_mismatch_k Rows outside the plan or of another width, or not one context
         *    per executor thread.
         */
        status_t insert(std::size_t first_row, nk::vector_view<key_t> keys, rows_view_t rows,
                        std::span<traversal_context_t> traversals) noexcept {
            std::size_t const count = rows.rows;
            if (!owner_) return status_t::uninitialized_k;
            if (!count) return status_t::success_k;
            if (keys.size() != count || first_row > batch_split_.batch_entries ||
                count > batch_split_.batch_entries - first_row || rows.columns != owner_->dimensions() ||
                traversals.size() != executor_->size())
                return status_t::config_mismatch_k;
            shared_lock_t const held {owner_->gate_};
            if (status_t const sized = owner_->size_traversals_(traversals, config_, 0); failed(sized)) return sized;
            if (ranges_.empty()) {
                std::lock_guard<fu::spin_mutex_t> const dispensing {owner_->dispensing_};
                for (std::size_t shard = 0; shard != claims_.size(); ++shard) {
                    index_t& shard_index = *owner_->shards_[batch_split_.open_shard_generation.first + shard].index;
                    auto const [range, dispense_status] = shard_index.dispense_(std::move(claims_[shard]));
                    if (failed(dispense_status)) {
                        // Closers go in shard order too, so every earlier range is empty.
                        for ([[maybe_unused]] iota_view<node_id_t> const before : ranges_)
                            assert(before.empty() && "A refused batch dispensed rows on an earlier shard");
                        ranges_.clear();
                        return dispense_status;
                    }
                    ranges_.push_back_reserved(range);
                }
            }
            // The key is assigned before the walk, so a row reachable once published reads a key.
            auto insert_row = [&](std::size_t task, std::size_t thread) noexcept {
                shard_node_t const target = shard_node_of_(first_row + task);
                index_t& shard_index = *owner_->shards_[target.shard].index;
                shard_index.assign_key_(target.node, keys[task]);
                shard_index.wire_(target.node, rows, task, traversals[thread], config_);
            };
            drain_in_order_(count, insert_row);
            inserted_entries_ += count;
            for (traversal_context_t const& traversal : traversals) {
                spent_.visited_members += traversal.visited_members;
                spent_.computed_distances += traversal.computed_distances;
                spent_.computed_distances_in_refines += traversal.computed_distances_in_refines;
                spent_.computed_distances_in_reverse_refines += traversal.computed_distances_in_reverse_refines;
                spent_.restarts += traversal.restarts;
                spent_.saturated_walks += traversal.saturated_walks;
            }
            return status_t::success_k;
        }

        /**
         *  @brief Publishes every row at once and reports the batch's telemetry. A plan with rows
         *      still to place refuses with @c config_mismatch_k and stays whole; a success leaves
         *      the plan spent.
         *
         *  Every touched shard's turn is awaited first, and while they are held one task per shard
         *  files its keys, the shard map's one writer; a batch spanning several shards then stores
         *  their watermarks inside a bracket of the commit sequence word, which readers retry
         *  around, so a lookup finds the batch's keys exactly when a search reaches its rows.
         *
         *  @par Statuses:
         *  - @c uninitialized_k The plan is empty or already spent.
         *  - @c config_mismatch_k Rows are still to place.
         */
        expected<insert_result_t> finish() noexcept {
            insert_result_t result = spent_;
            if (!owner_) return {result, status_t::uninitialized_k};
            result.new_size = owner_->size();
            if (inserted_entries_ != batch_split_.batch_entries) return {result, status_t::config_mismatch_k};
            shared_lock_t const held {owner_->gate_};
            std::size_t touched_shards = 0;
            for (iota_view<node_id_t> const range : ranges_) touched_shards += !range.empty();
            bool const spans_shards = touched_shards > 1;

            // No later range on a shard can take its turn ahead of this one, so the waits hold no bracket open.
            auto index_of = [&](std::size_t shard) noexcept -> index_t& {
                return *owner_->shards_[batch_split_.open_shard_generation.first + shard].index;
            };
            for (std::size_t shard = 0; shard != ranges_.size(); ++shard)
                if (!ranges_[shard].empty()) index_of(shard).await_turn_(ranges_[shard]);
            // Each held turn makes its task the shard map's one writer; a refused fan-out files
            // them here.
            auto publish_keys = [&](std::size_t shard, std::size_t, std::size_t) noexcept {
                if (!ranges_[shard].empty()) index_of(shard).publish_keys_(ranges_[shard]);
            };
            if (failed(executor_->for_tasks(ranges_.size(), publish_keys)))
                for (std::size_t shard = 0; shard != ranges_.size(); ++shard) publish_keys(shard, 0, 0);
            if (spans_shards) owner_->shards_.begin_commit();
            for (std::size_t shard = 0; shard != ranges_.size(); ++shard)
                if (!ranges_[shard].empty()) index_of(shard).publish_(ranges_[shard]);
            if (spans_shards) owner_->shards_.end_commit();

            result.new_size = owner_->size();
            owner_ = nullptr;
            return {result, status_t::success_k};
        }

      private:

        /** Where row @p row of the plan goes: its shard, and its ID inside that shard's range. */
        shard_node_t shard_node_of_(std::size_t row) const noexcept {
            entry_in_shard_t const target = batch_split_.shard_and_offset_of_entry(row);
            return {static_cast<u32_t>(batch_split_.open_shard_generation.first + target.shard),
                    static_cast<node_id_t>(ranges_[target.shard].first + target.offset)};
        }

        /** Runs @p per_row over @p rows rows taken in ascending order by the executor's threads, so
         *  a row's walk finds the rows of its range dispensed ahead of it mostly published; a
         *  refused fan-out runs them all on the calling thread, since a dispensed row must still be
         *  placed in order. */
        template <typename per_row_type_>
        void drain_in_order_(std::size_t rows, per_row_type_& per_row) noexcept {
            if (!rows) return;
            // Every row bumps it, so it takes its own line, away from the locals every thread reads.
            alignas(default_alignment_k) std::size_t cursor = 0;
            auto drain = [&](std::size_t thread, std::size_t) noexcept {
                atomic_ref<std::size_t> const next {cursor};
                for (std::size_t row = next.fetch_add(1, std::memory_order_relaxed); row < rows;
                     row = next.fetch_add(1, std::memory_order_relaxed))
                    per_row(row, thread);
            };
            if (failed(executor_->for_threads(drain))) drain(0, 0);
        }

        /** Every shard's capacity claim from the open shard generation's first in placement order,
         *  released by its own death until dispensed. */
        growing_buffer<shard_capacity_claim_t, forkunion_allocator_t> claims_;

        /** Every shard's range from the open shard generation's first in placement order, empty
         *  where no row goes; none before the first step. */
        growing_buffer<iota_view<node_id_t>, forkunion_allocator_t> ranges_;

        /** The executor's one-caller hold, kept while the plan's threads may run. */
        typename executor_t::claim_t executor_claim_;

        /** The executor the rows fan out over. */
        executor_t* executor_ = nullptr;

        /** The per-shard insertion configuration, applied uniformly. */
        insert_config_t config_ {};

        /** The batch split across the shard generation open at planning, which the claims and
         *  ranges start at, and the fresh ones it opened. */
        batch_split_t batch_split_ {};

        /** Rows placed so far. */
        std::size_t inserted_entries_ = 0;

        /** What every step's walks spent, which @ref finish reports. */
        insert_result_t spent_ {};

        /** The index inserted into; null in the empty plan a refused @ref plan_insert returns, and
         *  once @c finish published. */
        ram_indexes* owner_ = nullptr;

        friend ram_indexes;
    };

    /**
     *  @brief Opens a batched insertion of @p rows: takes the executor and every shard's share of
     *      capacity, all of them reversible.
     *
     *  Every fallible step happens here but the sizing of the contexts each step is handed, so the
     *  plan's rows can only be placed, and nothing is dispensed until the plan's first step. The
     *  rotation comes from the rows routed so far, which keeps a stream of small batches from
     *  drifting onto shard zero. A batch crossing the bound tops the open shard generation up and
     *  opens as many fresh shard generations as its other rows fill, so it is placed whole; a
     *  racing batch taking the same room first leaves this one's claim refused with @c overflow_k.
     *
     *  @param[in] rows Rows the plan places.
     *  @param[in,out] executor The threads, held until the plan dies.
     *  @param[in] config The per-shard insertion configuration, applied uniformly.
     *
     *  @par Statuses:
     *  - @c uninitialized_k No shard is placed.
     *  - @c contended_k A plan holds the index closed, another call holds @p executor, or a sealed
     *    set must open a shard generation.
     *  - @c overflow_k A racing batch took the room this one was routed into.
     */
    expected<insert_plan_t> plan_insert(std::size_t rows, executor_t& executor, insert_config_t config = {}) noexcept {
        if (shards_.empty()) return {{}, status_t::uninitialized_k};
        shared_lock_t const held {gate_};
        if (shards_.sealed()) return {{}, status_t::contended_k};
        insert_plan_t plan {shards_.allocator()};
        plan.executor_ = &executor;
        plan.config_ = config;
        plan.owner_ = this;
        if (!rows) return {std::move(plan), status_t::success_k};

        // One caller per executor; the steps walk the driver's contexts, so nothing here rewinds.
        auto [executor_claim, claim_status] = executor.claim();
        if (failed(claim_status)) return {{}, claim_status};
        plan.executor_claim_ = std::move(executor_claim);

        // The open shard generation takes what fits under the bound, fresh ones the rest, each
        // filled to it.
        auto const [batch_split, split_status] = shards_.split_batch_into_shards(
            rows, [this](shard_t const& like) noexcept { return build_shard_(like); });
        if (failed(split_status)) return {{}, split_status};
        plan.batch_split_ = batch_split;

        // A refusal releases every share already granted as the claims die.
        std::size_t const shards = batch_split.shards_spanned();
        if (failed(plan.claims_.reserve_exactly(shards)) || failed(plan.ranges_.reserve_exactly(shards)))
            return {{}, status_t::bad_alloc_k};
        for (std::size_t shard = 0; shard != shards; ++shard) {
            shard_t& slot = shards_[batch_split.open_shard_generation.first + shard];
            auto [shard_claim, status] = slot.index->claim(batch_split.entries_for_shard(shard));
            if (failed(status)) return {{}, status};
            plan.claims_.push_back_reserved(std::move(shard_claim));
        }
        return {std::move(plan), status_t::success_k};
    }

#pragma endregion Insertion

#pragma region Refinement

    /**
     *  @brief One refine in flight over every shard, holding every dispenser closed meanwhile.
     *
     *  It owns its shards' passes and part starts, so nothing about it lives in the index, and it
     *  walks on the contexts each call hands it; @ref run continues it after a cancel, and
     *  @ref finish heals what a stopped pass leaves and reopens. Every shard's @c refine_plan_t
     *  drains over the threads phase by phase, each thread pulling steps from its own shard first
     *  and stealing from the rest in the tail, so a pass costs one sweep of the graph however the
     *  shards are filled. An absorb's plan first moves its donors in, one after another, and only
     *  then opens the shards' passes over the parts they appended. A split's plan opens every
     *  shard's split at planning, the receivers' shards file the keys their parts received, and the
     *  phase boundary that publishes the parts empties this index inside its window. Every index
     *  the plan reads or moves closes at planning and stays closed until the plan dies, or for a
     *  donor until it moves in.
     */
    struct [[nodiscard]] refine_plan_t {
        refine_plan_t() noexcept : refine_plan_t(forkunion_allocator_t {}) {}

        /** An empty plan over @p allocator; only @ref plan_absorb and @ref plan_split fill it. */
        explicit refine_plan_t(forkunion_allocator_t allocator) noexcept
            : shard_plans_(allocator), cursors_(allocator), part_bases_(allocator), part_offsets_(allocator),
              moves_(allocator), receiver_shards_(allocator) {}
        refine_plan_t(refine_plan_t&& other) noexcept
            : shard_plans_(std::move(other.shard_plans_)), cursors_(std::move(other.cursors_)),
              part_bases_(std::move(other.part_bases_)), part_offsets_(std::move(other.part_offsets_)),
              moves_(std::move(other.moves_)), donors_(std::exchange(other.donors_, {})),
              absorbed_donors_(other.absorbed_donors_), receivers_(other.receivers_),
              closed_receivers_(std::exchange(other.closed_receivers_, 0)),
              receiver_shards_(std::move(other.receiver_shards_)), config_(other.config_), visited_(other.visited_),
              population_(other.population_), owner_(std::exchange(other.owner_, nullptr)) {}
        refine_plan_t& operator=(refine_plan_t&&) = delete;
        refine_plan_t(refine_plan_t const&) = delete;
        refine_plan_t& operator=(refine_plan_t const&) = delete;

        /** An unfinished plan heals nothing: it reopens, and the graph is as the last step left
         *  it. */
        ~refine_plan_t() noexcept { release_(); }

        /**
         *  @brief Runs the pending phases on @p executor until the pass ends or @p on_visited
         *      cancels; @c cancelled_k means a later call continues exactly where this one stopped.
         *
         *  Moves the donors still pending in first, as @ref plan_absorb describes; a failing one
         *  stays pending and closed, with every later one, for the next call, and a moved one stays
         *  empty, closed and sealed. The executor is held only for the call.
         *
         *  @param[in,out] executor The executor the plan was opened over.
         *  @param[in,out] traversals The caller's context of every executor thread, which the walks
         *      run on; the call sizes them.
         *  @param[in] on_visited Fires per re-selected step with the nodes visited so far and the
         *      population; a @c cancel_k answer stops the drain where its steps ran.
         *
         *  @par Statuses:
         *  - @c uninitialized_k The plan is bound to no index, or @p executor holds no pool or no
         *    workers.
         *  - @c config_mismatch_k @p traversals holds other than one context per thread, or a
         *    move step meets rows outside its plan.
         *  - @c contended_k Another call holds @p executor, its pool runs another dispatch, or a
         *    donor stopped emptying under the move.
         *  - @c overflow_k A move passes a shard's rows, or a queue capacity wraps when rounded.
         */
        template <typename progress_type_ = dummy_progress_t>
        status_t run(executor_t& executor, std::span<traversal_context_t> traversals,
                     progress_type_&& on_visited = {}) noexcept {
            if (!owner_) return status_t::uninitialized_k;
            if (executor.size() != traversals.size()) return status_t::config_mismatch_k;
            auto [executor_claim, claim_status] = executor.claim();
            if (failed(claim_status)) return claim_status;
            for (; absorbed_donors_ != donors_.size(); ++absorbed_donors_) {
                ram_indexes*& donor = donors_[absorbed_donors_];
                if (status_t const absorbed = owner_->absorb_(*donor, moves_of_(donors_.subspan(absorbed_donors_)),
                                                              executor, *this);
                    failed(absorbed))
                    return absorbed;
                donor = nullptr;
            }
            if (shard_plans_.empty())
                if (status_t const opened = open_shards_(); failed(opened)) return opened;
            // Only once the donors moved in does every shard have the geometry the walks size by.
            if (status_t const sized = size_walks_(traversals); failed(sized)) return sized;
            return drain_(executor, traversals, on_visited);
        }

        /**
         *  @brief Stops any re-selection, audits and heals on the caller's @p traversals, one per
         *      executor thread, reopens the dispensers, and reports. A plan whose donors have not
         *      all moved in heals nothing and reports nothing.
         *
         *  @par Statuses:
         *  - @c uninitialized_k The plan is bound to no index, or @p executor holds no pool or no
         *    workers.
         *  - @c config_mismatch_k @p traversals holds other than one context per thread.
         *  - @c contended_k Another call holds @p executor, or its pool runs another dispatch.
         *  - @c overflow_k A queue capacity wraps when rounded.
         */
        expected<refine_report_t> finish(executor_t& executor, std::span<traversal_context_t> traversals) noexcept {
            if (!owner_) return {{}, status_t::uninitialized_k};
            if (executor.size() != traversals.size()) return {{}, status_t::config_mismatch_k};
            auto [executor_claim, claim_status] = executor.claim();
            if (failed(claim_status)) return {{}, claim_status};
            status_t healed = status_t::success_k;
            if (!shard_plans_.empty()) {
                // The steps a stopped re-selection left go unrun; a stopped heal phase continues.
                for (std::size_t shard = 0; shard != shard_plans_.size(); ++shard) {
                    if (!shard_plans_[shard].reselecting()) continue;
                    shard_plans_[shard].stop();
                    static_cast<void>(shard_plans_[shard].advance());
                    cursors_[shard] = 0;
                }
                dummy_progress_t heal_alone;
                healed = size_walks_(traversals);
                if (succeeded(healed)) healed = drain_(executor, traversals, heal_alone);
            }
            refine_report_t const result = running();
            release_();
            return {result, healed};
        }

        /** The totals so far, summed over the shards' passes between calls: what @ref finish
         *  reports once it healed, and zeros while donors are pending. */
        refine_report_t running() const noexcept {
            refine_report_t result;
            for (typename index_t::refine_plan_t const& plan : shard_plans_) result += plan.finish().value;
            return result;
        }

      private:
        friend ram_indexes;
        template <typename>
        friend struct ram_compact_plan;
        template <typename, typename>
        friend struct vram_nvidia_compact_plan;
        template <typename, typename>
        friend struct vram_amd_compact_plan;

        /** The moves of the first of @p pending sources: sources move in order, so the shards of
         *  every later one, this index aside, close the moves. */
        std::span<shard_move_t const> moves_of_(std::span<ram_indexes* const> pending) const noexcept {
            std::size_t const count = pending.front()->shards_.size();
            std::size_t later = 0;
            for (ram_indexes const* source : pending.subspan(1))
                if (source != owner_) later += source->shards_.size();
            return to_span(moves_).subspan(moves_.size() - later - count, count);
        }

        /** Reopens this index, every donor still pending and every receiver it closed; the plan is
         *  unbound after, so a finished plan's destructor releases nothing twice. */
        void release_() noexcept {
            if (owner_) std::exchange(owner_, nullptr)->shards_.reopen();
            for (ram_indexes* pending : donors_.subspan(std::exchange(absorbed_donors_, 0))) pending->shards_.reopen();
            donors_ = {};
            for (; closed_receivers_; --closed_receivers_) receivers_[closed_receivers_ - 1]->shards_.reopen();
        }

        /** Sizes the caller's @p traversals for the passes' walks. */
        status_t size_walks_(std::span<traversal_context_t> traversals) noexcept {
            // A split walks its receivers, which keep the geometry once this index empties.
            ram_indexes& walked = receivers_.empty() ? *owner_ : *receivers_[0];
            return walked.size_traversals_(traversals, insert_config_t {config_.expansion_factor}, 0);
        }

        /** Opens every shard's pass over its parts, once no donor is pending; the shards are the
         *  ones closed at planning, and a shard generation opened since takes no part. */
        status_t open_shards_() noexcept {
            for (std::size_t shard = 0; shard != cursors_.size(); ++shard) {
                std::span<node_id_t const> parts;
                // A refine re-selects every row; an absorb only the rows appended after the
                // resident part.
                iota_view<node_id_t> nodes {0, node_id_unknown_k};
                if (!part_offsets_.empty()) {
                    parts = {part_bases_.data() + part_offsets_[shard],
                             part_offsets_[shard + 1] - part_offsets_[shard]};
                    nodes.first = parts.size() > 1 ? parts[1] : node_id_unknown_k;
                }
                auto [plan, status] = owner_->shards_[shard].index->plan_refine(nodes, {}, parts, config_);
                if (failed(status)) return shard_plans_.clear(), status;
                shard_plans_.push_back_reserved(std::move(plan));
            }
            for (typename index_t::refine_plan_t const& plan : shard_plans_) population_ += plan.planned_nodes();
            return status_t::success_k;
        }

        /** Drains every shard's pending phases from where the cursors stand, advancing each plan
         *  between them, until none follows or a step fails or cancels. */
        template <typename progress_type_>
        status_t drain_(executor_t& executor, std::span<traversal_context_t> traversals,
                        progress_type_& on_visited) noexcept {
            // Every step bumps it, so it takes its own line, away from the locals workers read.
            alignas(default_alignment_k) std::size_t visited = visited_;
            auto run_step = [&](std::size_t shard, typename index_t::refine_plan_t::step_t const& step,
                                std::size_t thread, nk::matrix_span<node_id_t>) noexcept -> status_t {
                typename index_t::refine_plan_t& plan = shard_plans_[shard];
                bool const reselecting = plan.reselecting();
                if (status_t const ran = plan.run(step, traversals[thread]); failed(ran)) return ran;
                if (!reselecting) return status_t::success_k;
                std::size_t const done = atomic_ref<std::size_t>(visited).fetch_add(step.nodes.size(),
                                                                                    std::memory_order_relaxed) +
                                         step.nodes.size();
                return on_visited(done, population_) == continuation_t::cancel_k ? status_t::cancelled_k
                                                                                 : status_t::success_k;
            };
            std::span<std::size_t> const cursors = to_writable_span(cursors_);
            status_t drained = status_t::success_k;
            std::size_t advancing = 0;
            do {
                // Every claimed step ran, so a stopped drain's cursors say where the next starts.
                drained = owner_->drain_steps_(shard_plans_, cursors, executor, 0, run_step);
                if (failed(drained)) break;
                // The boundary publishing a split's parts shows a search every receiver shard's
                // rows and keys or none, and forgets this index's inside its window, so the gate is
                // taken before anything publishes.
                bool const moving = !receivers_.empty() && shard_plans_[0].moving();
                unique_lock_t emptying;
                if (moving) {
                    emptying = unique_lock_t {owner_->gate_, std::try_to_lock};
                    if (!emptying) {
                        drained = status_t::contended_k;
                        break;
                    }
                    owner_->shards_.begin_window();
                    for (ram_indexes* receiver : receivers_) receiver->shards_.begin_commit();
                }
                advancing = 0;
                for (typename index_t::refine_plan_t& plan : shard_plans_) advancing += plan.advance();
                if (moving) {
                    for (ram_indexes* receiver : receivers_) receiver->shards_.end_commit();
                    if (!shard_plans_[0].moving()) owner_->clear_();
                    owner_->shards_.end_window();
                }
                std::memset(cursors.data(), 0, cursors.size_bytes());
            } while (advancing);
            visited_ = visited;
            return drained;
        }

        /** Every shard's own pass in placement order; none until the donors moved in. */
        growing_buffer<typename index_t::refine_plan_t, forkunion_allocator_t> shard_plans_;

        /** One step cursor per shard into its current phase. */
        growing_buffer<std::size_t, forkunion_allocator_t> cursors_;

        /** The parts the moves appended, ascending per shard, @c node_id_unknown_k past the
         *  last. */
        growing_buffer<node_id_t, forkunion_allocator_t> part_bases_;

        /** Where each shard's run of @c part_bases_ starts, and one past the last; empty with no
         *  donors. */
        growing_buffer<std::size_t, forkunion_allocator_t> part_offsets_;

        /** One move per source shard, in move order, as @c plan_moves_ packed them;
         *  empty with no donors. */
        growing_buffer<shard_move_t, forkunion_allocator_t> moves_;

        /** The caller's donor slots, each closed at planning and nulled once moved in. */
        std::span<ram_indexes*> donors_;

        /** Donors moved in so far. */
        std::size_t absorbed_donors_ = 0;

        /** The caller's receiver slots of a split, one part each; empty otherwise. */
        std::span<ram_indexes*> receivers_;

        /** Receivers whose dispensers the split closed, in receiver order. */
        std::size_t closed_receivers_ = 0;

        /** Per shard, the receivers' same shards in receiver order - what each shard's split
         *  scatters into. */
        growing_buffer<index_t*, forkunion_allocator_t> receiver_shards_;

        /** The passes' knobs. */
        refine_config_t config_ {};

        /** Nodes re-selected across every call so far. */
        std::size_t visited_ = 0;

        /** The population @c on_visited reports: the nodes every shard's re-selection rounds step
         *  over once the donors moved in, or a split's rows. */
        std::size_t population_ = 0;

        /** The index refined, closed while the plan lives; null in an empty plan and once
         *  finished. */
        ram_indexes* owner_ = nullptr;
    };

    /** Opens a refine of every shard: closes the dispensers and opens the shards' passes, as
     *  @ref plan_absorb does with no donors. Re-selection claims no identity and moves no
     *  watermark, so searches run throughout.
     *
     *  @par Statuses:
     *  - @c uninitialized_k No shard is placed, or @p executor has no
     *    threads.
     *  - @c contended_k A call holds the gate, or a batch or plan holds a shard's dispenser.
     */
    expected<refine_plan_t> plan_refine(refine_config_t config, executor_t const& executor) noexcept {
        return plan_absorb({}, config, executor);
    }

    /**
     *  @brief Re-selects the neighbors of every published node across the shards, then heals
     *      the islands that leaves: @ref plan_refine, then @c run and @c finish.
     *
     *  Re-selection claims no identity and moves no watermark, so searches run throughout and a
     *  stopped pass leaves the graph correct. Every shard's dispenser stays closed for the pass, so
     *  a batch in flight refuses it with @c contended_k, and an insertion arriving meanwhile is
     *  refused.
     *
     *  @param[in] config Optional tuning knobs; zeros mean "derive from connectivity".
     *  @param[in,out] executor The threads and their scratch.
     *  @param[in] on_visited Fires per re-selected step with the nodes visited so far and the
     *      population; a @c cancel_k answer skips the rest of the re-selection, still heals, and
     *      reports @c cancelled_k.
     *
     *  @par Statuses:
     *  - @c uninitialized_k No shard is placed, or @p executor holds no
     *    threads, pool or arenas.
     *  - @c contended_k A call holds the gate, a batch or plan holds a shard's dispenser, another
     *    call holds @p executor, or its pool runs another dispatch.
     *  - @c overflow_k A queue capacity wraps when rounded.
     */
    template <typename progress_type_ = dummy_progress_t>
    expected<refine_report_t> refine(refine_config_t config, executor_t& executor,
                                     progress_type_&& on_visited = {}) noexcept {
        auto [plan, status] = plan_refine(config, executor);
        if (failed(status)) return {{}, status};
        return drive_refine_(plan, executor, on_visited);
    }

    /**
     *  @brief Opens an absorb of @p donors, whose plan's first phase moves every row of each into
     *      this index, one donor after another: donor shards, largest first, each append to the
     *      shard with the fewest rows that keeps it under @ref shard_capacity, one on its own
     *      memory domain on ties, with its edges rebased and its keys beside it, as a part of its
     *      own.
     *
     *  Each donor moves in through one plan that does every fallible thing first: the donor's gate
     *  and a load claiming all the rows each receiving shard takes. Its chunk steps then drain over
     *  the threads, each copying the donor's chunks straight into the shard's own, keys beside
     *  them, and cannot fail; one task per receiving shard files its keys, and every shard's rows
     *  publish inside one commit bracket, so a search and a lookup see all of them or none. The
     *  donor then forgets its rows in place inside its window, keeping its shards and their
     *  capacity, its slot in @p donors nulls, it stays closed and sealed, consumed, and its shards
     *  become parts of the receiving shards in the plan. A shard empty beforehand keeps the first
     *  part moving into it as its resident part.
     *
     *  Every fallible step of every move happens here or in the move's own plan per donor, never in
     *  a step. Once the last donor moved in, the plan's re-selection walks the rows the moves
     *  appended across the parts, round by round, so a group costs one refine rather than one per
     *  donor; a cancel pauses only that walk.
     *
     *  Searches of this index and of the donors run throughout. This index and every donor close
     *  here, each under its own gate, so the packing never goes stale: this index stays closed
     *  while the plan lives and each donor until it moves in, a batch in flight or holding claims
     *  on any of them refuses the plan with @c contended_k, and an insertion into any of them
     *  arriving meanwhile is refused.
     *
     *  @param[in,out] donors The indexes giving their rows up, each nulled once consumed; the slots
     *      must outlive the moves.
     *  @param[in] config The refine's knobs; zeros mean "derive from connectivity".
     *  @param[in] executor The executor every @c run and @c finish names.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k A null donor, another metric or geometry, or this index itself.
     *  - @c contended_k A call or batch in flight on this index or a donor, a repeated donor, or
     *    another refine, absorb, split, compaction or load holding either.
     *  - @c unsupported_k A donor shard past @ref shard_capacity, which would have to be cut.
     */
    expected<refine_plan_t> plan_absorb(std::span<ram_indexes*> donors, refine_config_t config,
                                        executor_t const& executor) noexcept {
        std::size_t const shards = shards_.size(), threads = executor.size();
        if (!shards || !threads) return {{}, status_t::uninitialized_k};
        for (ram_indexes const* donor : donors)
            if (!donor || donor == this || donor->kernel() != kernel() || donor->geometry() != geometry_)
                return {{}, status_t::config_mismatch_k};
        if (status_t const closed = close_gated_(); failed(closed)) return {{}, closed};
        refine_plan_t plan {shards_.allocator()};
        plan.owner_ = this;
        plan.config_ = config;
        for (std::size_t closing = 0; closing != donors.size(); ++closing)
            if (status_t const closed = donors[closing]->close_gated_(); failed(closed)) {
                while (closing) donors[--closing]->shards_.reopen();
                return {{}, closed};
            }
        plan.donors_ = donors;
        if (!donors.empty())
            if (status_t const packed = plan_moves_(donors, shards_.shard_capacity(), rank_fewest_entries, plan);
                failed(packed))
                return {{}, packed};
        // The moves may have opened a shard generation, whose shards take part like the others.
        std::size_t const receiving_shards = donors.empty() ? shards : plan.part_offsets_.size() - 1;
        if (failed(plan.shard_plans_.reserve_exactly(receiving_shards)) ||
            failed(plan.cursors_.resize(receiving_shards)))
            return {{}, status_t::bad_alloc_k};
        if (donors.empty())
            if (status_t const opened = plan.open_shards_(); failed(opened)) return {{}, opened};
        return {std::move(plan), status_t::success_k};
    }

    /**
     *  @brief Moves every row of each of @p donors into this index and links them in:
     *      @ref plan_absorb, then @c run and @c finish. @p on_visited answers as in @ref refine,
     *      for the re-selection alone; a cancel still heals and reports @c cancelled_k.
     *
     *  @par Statuses:
     *  - @c uninitialized_k No shard is placed, or @p executor holds no threads, pool or arenas.
     *  - @c config_mismatch_k A null donor, this index itself, another kernel or geometry, or a
     *    move step meeting rows outside its plan.
     *  - @c contended_k A call or plan holds this index or a donor, another call holds
     *    @p executor, its pool runs another dispatch, or a donor stopped emptying.
     *  - @c unsupported_k A donor shard is past @ref shard_capacity.
     *  - @c overflow_k No shard takes a move, a move passes a shard's rows, or a queue
     *    capacity wraps.
     */
    template <typename progress_type_ = dummy_progress_t>
    expected<refine_report_t> absorb(std::span<ram_indexes*> donors, refine_config_t config, executor_t& executor,
                                     progress_type_&& on_visited = {}) noexcept {
        auto [plan, status] = plan_absorb(donors, config, executor);
        if (failed(status)) return {{}, status};
        return drive_refine_(plan, executor, on_visited);
    }

    /**
     *  @brief Moves every row of @p donor into this index, as the many-donor @ref absorb does when
     *      handed one donor alone.
     *
     *  @par Statuses:
     *  - @c uninitialized_k No shard is placed, or @p executor holds no
     *    threads, pool or arenas.
     *  - @c config_mismatch_k @p donor is this index itself, of another kernel or geometry, or a
     *    move step meets rows outside its plan.
     *  - @c contended_k A call or plan holds this index or @p donor, another call holds
     *    @p executor, its pool runs another dispatch, or @p donor stopped emptying.
     *  - @c unsupported_k A donor shard is past @ref shard_capacity.
     *  - @c overflow_k No shard takes a move, a move passes a shard's rows, or a queue
     *    capacity wraps.
     */
    expected<refine_report_t> absorb(ram_indexes& donor, refine_config_t config, executor_t& executor) noexcept {
        ram_indexes* donors[1] = {&donor};
        return absorb(std::span<ram_indexes*>(donors), config, executor);
    }

    /**
     *  @brief Opens a split of this index into @p receivers - the inverse of @c plan_absorb:
     *      shard @c s of this index splits into shard @c s of every receiver, one graph-local part
     *      per receiver, with its keys beside its rows, and this index empties.
     *
     *  Every shard's split opens here, as @c index_t::plan_split describes, and so does every other
     *  fallible thing: this index and every receiver closed, each under its own gate. The plan's
     *  first phases then partition and scatter every shard over the threads, each key going beside
     *  its row, one step per receiver shard files its keys, and the boundary publishing the parts
     *  shows a search and a lookup every receiver shard's rows or none, while this index forgets
     *  its rows in place inside its window, keeping its shards and their capacity. Every receiver
     *  shard then re-selects the rows that lost a link and heals; a cancel pauses only that walk.
     *
     *  Searches of this index and of the receivers run throughout, a search of this index retrying
     *  around the window that empties it; an insertion into any of them arriving meanwhile is
     *  refused.
     *
     *  @param[in] receivers The empty indexes taking the parts, at least two, all distinct, each
     *      with as many shards as this index; the slots must outlive the plan.
     *  @param[in] config The refine's knobs, zeros meaning "derive from connectivity", and the seed
     *      picking every shard's part roots: equal seeds over equal graphs split alike.
     *  @param[in] executor The executor every @c run and @c finish names.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k Fewer than two receivers, a null, repeated or non-empty one, this
     *    index itself, another metric, geometry or shard count.
     *  - @c contended_k A call or batch in flight on this index or a receiver, or another
     *    refine, absorb, split, compaction or load holding either.
     */
    expected<refine_plan_t> plan_split(std::span<ram_indexes*> receivers, split_config_t config,
                                       executor_t const& executor) noexcept {
        std::size_t const shards = shards_.size(), threads = executor.size(), parts = receivers.size();
        if (!shards || !threads) return {{}, status_t::uninitialized_k};
        if (parts < 2) return {{}, status_t::config_mismatch_k};
        for (std::size_t receiver = 0; receiver != parts; ++receiver) {
            ram_indexes const* into = receivers[receiver];
            if (!into || into == this || into->kernel() != kernel() || into->shards_.size() != shards ||
                into->geometry() != geometry_)
                return {{}, status_t::config_mismatch_k};
            for (std::size_t earlier = 0; earlier != receiver; ++earlier)
                if (receivers[earlier] == into) return {{}, status_t::config_mismatch_k};
        }
        if (status_t const closed = close_gated_(); failed(closed)) return {{}, closed};
        refine_plan_t plan {shards_.allocator()};
        plan.owner_ = this;
        plan.receivers_ = receivers;
        plan.config_ = config.refine;
        // A shard generation opened before the close would hold rows no shard's split moves.
        if (shards_.size() != shards) return {{}, status_t::contended_k};
        // Closed, a receiver can take no row, so its emptiness holds for the plan's life.
        for (ram_indexes* into : receivers) {
            if (status_t const closed = into->close_gated_(); failed(closed)) return {{}, closed};
            ++plan.closed_receivers_;
            if (into->size()) return {{}, status_t::config_mismatch_k};
        }
        if (failed(plan.shard_plans_.reserve_exactly(shards)) || failed(plan.cursors_.resize(shards)) ||
            failed(plan.receiver_shards_.reserve_exactly(shards * parts)))
            return {{}, status_t::bad_alloc_k};
        for (std::size_t shard = 0; shard != shards; ++shard)
            for (ram_indexes* into : receivers)
                plan.receiver_shards_.push_back_reserved(into->shards_[shard].index.get());
        for (std::size_t shard = 0; shard != shards; ++shard) {
            std::span<index_t*> const into {plan.receiver_shards_.data() + shard * parts, parts};
            auto [split, status] = shards_[shard].index->plan_split(into, {config.refine, config.seed + shard});
            if (failed(status)) return {{}, status};
            plan.shard_plans_.push_back_reserved(std::move(split));
        }
        plan.population_ = size();
        return {std::move(plan), status_t::success_k};
    }

    /** Moves every row of this index into one graph-local part per receiver and links each in:
     *  @ref plan_split, then @c run and @c finish. @p on_visited answers as in @ref refine, for the
     *  re-selection alone; a cancel still heals and reports @c cancelled_k.
     *
     *  @par Statuses:
     *  - @c uninitialized_k No shard is placed, or @p executor holds no
     *    threads, pool or arenas.
     *  - @c config_mismatch_k Fewer than two receivers, a null, repeated or non-empty one, this
     *    index itself, another kernel, geometry or shard count.
     *  - @c contended_k A call or plan holds this index or a receiver, a shard generation opened
     *    meanwhile, another call holds @p executor, or its pool runs another dispatch.
     *  - @c overflow_k A part passes a receiver's ceiling, or a queue capacity wraps.
     */
    template <typename progress_type_ = dummy_progress_t>
    expected<refine_report_t> split(std::span<ram_indexes*> receivers, split_config_t config, executor_t& executor,
                                    progress_type_&& on_visited = {}) noexcept {
        auto [plan, status] = plan_split(receivers, config, executor);
        if (failed(status)) return {{}, status};
        return drive_refine_(plan, executor, on_visited);
    }

    /**
     *  @brief Reachability of every shard from as many landmarks as one search draws, swept level
     *      by level over the threads. Read-only, so concurrent insertions leave fresh nodes
     *      uncounted; bracketed by the epoch as every reader is.
     *
     *  @param[in,out] executor The threads the levels sweep on; each shard's bitsets come from the
     *      shard's own allocator.
     *
     *  @par Statuses:
     *  - @c uninitialized_k No shard is placed, or @p executor holds no pool or no workers.
     *  - @c contended_k Another call holds @p executor, its pool runs another dispatch, or two
     *    destructive windows overlapped the audit.
     */
    expected<reachability_t> health(executor_t& executor) noexcept {
        if (shards_.empty()) return {{}, status_t::uninitialized_k};
        auto [executor_claim, claim_status] = executor.claim();
        if (failed(claim_status)) return {{}, claim_status};
        return shards_.read_retrying_once([&]() noexcept { return health_(executor); });
    }

#pragma endregion Refinement

#pragma region Search

    /**
     *  @brief Scatters one query over every shard and gathers the rankings into one, ascending by
     *      distance - the end-to-end multi-shard lookup.
     *
     *  @param[in] query_vector The query, in the shards' stored scalar type.
     *  @param[in,out] keys Keys of the merged ranking; its length is the capacity. Resolved through
     *      the shards' keys, so colliding ID spaces still merge into one ranking.
     *  @param[in,out] distances Distances of the same ranking, parallel to @p keys.
     *  @param[in] config The per-shard search configuration, applied uniformly.
     *  @param[in,out] traversal The calling thread's traversal context, reused per shard.
     *
     *  A failing shard stops the scatter but keeps whatever earlier shards already merged, so the
     *  reported count and the ranking always agree. Every shard is walked below one consistent
     *  snapshot of the watermarks, so a batch is ranked whole or not at all, and a destructive
     *  window overlapping the walk reruns it once, then answers @c contended_k.
     *
     *  @par Statuses:
     *  - @c unsupported_k @p config names a traversal other than a walk.
     *  - @c config_mismatch_k The query's width differs from the dimensions.
     *  - @c overflow_k A queue capacity of @p traversal wraps when rounded.
     *  - @c contended_k Two destructive windows overlapped the walk.
     */
    expected<search_result_t> search(vector_view_t query_vector, std::span<key_t> keys, std::span<distance_t> distances,
                                     search_config_t config, traversal_context_t& traversal) noexcept {
        return shards_.read_retrying_once(
            [&]() noexcept { return search_(query_vector, traversal, config, keys, distances); });
    }

    /**
     *  @brief Scatters a whole matrix of queries over every shard, one thread per query in flight -
     *      the executor-driven sibling of @ref search, taking the executor exactly where the
     *      single form takes a context.
     *
     *  @param[in] queries The batch, one query per row, in any scalar NumKong casts into the stored
     *      one; each thread imports its own.
     *  @param[out] matches Merged rankings, one query per row, as many matches as their width:
     *      keys, scores as @p score_type_, and per-query hit counts.
     *  @param[in] config The per-shard search configuration, applied uniformly; a
     *      @c traversal_t::scan_k scan scores every published row exactly instead, ignoring the
     *      breadth.
     *  @param[in,out] executor The threads; one query occupies one at a time.
     *  @param[in,out] traversals One context per executor thread, carved by the caller; the call
     *      reseeds and sizes each, and reports only what it spent.
     *
     *  A thread that fails stops its own query and leaves the rest to finish; the first failure is
     *  what the merged result reports, and every other query's outputs stand.
     *
     *  @par Statuses:
     *  - @c uninitialized_k No shard is placed, or @p executor holds no pool
     *    or no workers.
     *  - @c config_mismatch_k The counts or rankings disagree with the batch, @p traversals holds
     *    other than one context per thread, or the queries' width differs from the dimensions.
     *  - @c unsupported_layout_k A ranking row is not contiguous.
     *  - @c unsupported_k @p config names neither a walk nor a scan.
     *  - @c contended_k Another call holds @p executor, its pool runs another dispatch, or two
     *    destructive windows overlapped a query.
     *  - @c overflow_k A queue capacity wraps when rounded.
     */
    template <typename score_type_>
    expected<search_result_t> search(rows_view_t queries, ranked_matches_span<key_t, score_type_> matches,
                                     search_config_t config, executor_t& executor,
                                     std::span<traversal_context_t> traversals) noexcept {
        return search_if(queries, matches, config, dummy_predicate_t {}, executor, traversals);
    }

    /**
     *  @brief The batched @ref search under a lightning indexer, sharing its whole contract: every
     *      distance is 1 − its key's indexer score against the heads of its query row.
     *
     *  @param[in] queries The batch, as the batched @ref search takes it.
     *  @param[in] heads The indexer heads of every query row.
     *  @param[out] matches Rankings, as the batched @ref search writes them.
     *  @param[in] config The per-shard walk configuration, applied uniformly.
     *  @param[in,out] executor The threads, as the batched @ref search takes them.
     *  @param[in,out] traversals One context per executor thread, as the batched @ref search takes.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k The shards do not walk by dot.
     */
    expected<search_result_t> search(rows_view_t queries, indexer_heads<scalar_t> const& heads,
                                     ranked_matches_span<key_t, distance_t> matches, search_config_t config,
                                     executor_t& executor, std::span<traversal_context_t> traversals) noexcept
        requires floating_scalar<scalar_t>
    {
        if (kernel() != nk_kernel_dot_k) return {{}, status_t::config_mismatch_k};
        return search_rows_(queries, heads, executor, traversals, config, matches.keys, matches.scores, matches.counts,
                            dummy_predicate_t {});
    }

    /**
     *  @brief The filtered sibling of the batched @ref search, sharing its whole contract.
     *
     *  @param[in] queries The batch, as the batched @ref search takes it.
     *  @param[out] matches Rankings, as the batched @ref search writes them.
     *  @param[in] config The per-shard walk configuration, applied uniformly.
     *  @param[in] predicate The per-key admission predicate; @c dummy_predicate_t admits everything
     *      and compiles the predicated machinery out.
     *  @param[in,out] executor The threads, as the batched @ref search takes them.
     *  @param[in,out] traversals One context per executor thread, as the batched @ref search takes.
     *
     *  Shards test node IDs, so each one's keys column translates before the predicate sees a
     *  candidate. A scan asks it only about rows closer than the worst match their query already
     *  holds, so a selective predicate is called far less often than rows are scored.
     *
     *  @par Statuses:
     *  - @c uninitialized_k No shard is placed, or @p executor holds no pool or no workers.
     *  - @c config_mismatch_k The counts or rankings disagree with the batch, @p traversals holds
     *    other than one context per thread, or the queries' width differs from the dimensions.
     *  - @c unsupported_layout_k A ranking row is not contiguous.
     *  - @c unsupported_k @p config names neither a walk nor a scan.
     *  - @c contended_k Another call holds @p executor, its pool runs another dispatch, or two
     *    destructive windows overlapped a query.
     *  - @c overflow_k A queue capacity wraps when rounded.
     */
    template <typename score_type_, typename predicate_type_>
    expected<search_result_t> search_if(rows_view_t queries, ranked_matches_span<key_t, score_type_> matches,
                                        search_config_t config, predicate_type_&& predicate, executor_t& executor,
                                        std::span<traversal_context_t> traversals) noexcept {
        static_assert((!aggregating_predicate<std::remove_cvref_t<predicate_type_>, key_t, walk_payload_t> &&
                       !aggregating_predicate<std::remove_cvref_t<predicate_type_>, key_t, scan_payload_t>) ||
                          per_thread_predicates<std::remove_cvref_t<predicate_type_>>,
                      "An engine driving its own threads takes an aggregating predicate per thread");
        if (config.traversal == traversal_t::scan_k)
            return scan_ranked_(queries, executor, traversals, config, matches.keys, matches.scores, matches.counts,
                                predicate);
        return search_rows_(queries, dummy_heads_t {}, executor, traversals, config, matches.keys, matches.scores,
                            matches.counts, predicate);
    }

    /** The filtered sibling for one aggregating predicate per thread, each gathering a walk's
     *  @c walk_payload_t, so it serves walks alone and a scan answers @c config_mismatch_k.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k @p config names a scan.
     *  - @c uninitialized_k No shard is placed, or @p executor holds no pool
     *    or no workers.
     *  - @c config_mismatch_k The counts or rankings disagree with the batch, @p traversals holds
     *    other than one context per thread, or the queries' width differs from the dimensions.
     *  - @c unsupported_layout_k A ranking row is not contiguous.
     *  - @c unsupported_k @p config names neither a walk nor a scan.
     *  - @c contended_k Another call holds @p executor, its pool runs another dispatch, or two
     *    destructive windows overlapped a query.
     *  - @c overflow_k A queue capacity wraps when rounded.
     */
    template <typename score_type_, typename predicate_type_>
        requires aggregating_predicate<typename std::remove_cvref_t<predicate_type_>::thread_predicate_t, key_t,
                                       walk_payload_t>
    expected<search_result_t> search_if(rows_view_t queries, ranked_matches_span<key_t, score_type_> matches,
                                        search_config_t config, predicate_type_&& predicate, executor_t& executor,
                                        std::span<traversal_context_t> traversals) noexcept {
        if (config.traversal == traversal_t::scan_k) return {{}, status_t::config_mismatch_k};
        return search_rows_(queries, dummy_heads_t {}, executor, traversals, config, matches.keys, matches.scores,
                            matches.counts, predicate);
    }

    /** The filtered sibling for one aggregating predicate per thread, each gathering a scan's
     *  @c scan_payload_t, so it serves scans alone and a walk answers @c config_mismatch_k.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k @p config names a walk.
     *  - @c uninitialized_k No shard is placed, or @p executor holds no pool
     *    or no workers.
     *  - @c config_mismatch_k The counts or rankings disagree with the batch, @p traversals holds
     *    other than one context per thread, or the queries' width differs from the dimensions.
     *  - @c unsupported_layout_k A ranking row is not contiguous.
     *  - @c contended_k Another call holds @p executor, its pool runs another dispatch, or two
     *    destructive windows overlapped a block.
     */
    template <typename score_type_, typename predicate_type_>
        requires aggregating_predicate<typename std::remove_cvref_t<predicate_type_>::thread_predicate_t, key_t,
                                       scan_payload_t>
    expected<search_result_t> search_if(rows_view_t queries, ranked_matches_span<key_t, score_type_> matches,
                                        search_config_t config, predicate_type_&& predicate, executor_t& executor,
                                        std::span<traversal_context_t> traversals) noexcept {
        if (config.traversal != traversal_t::scan_k) return {{}, status_t::config_mismatch_k};
        return scan_ranked_(queries, executor, traversals, config, matches.keys, matches.scores, matches.counts,
                            predicate);
    }

    /**
     *  @brief Graph products of a whole matrix of left-hand rows against every shard: each row keeps
     *      the pairs its walks scored, keys resolved and dots cast into the products' scalar, or
     *      under a @c traversal_t::scan_k scan the largest exact dots, largest first.
     *
     *  @param[in] left The left-hand rows, one walk per row, in any scalar NumKong casts into the
     *      stored one; each thread imports its own.
     *  @param[out] products The tape the kept pairs land on, one run per left-hand row from its
     *      offset, each up to the tape's stride of pairs: each pair's key and its dot as
     *      @p product_type_. Each row's thread stages its run at the row's stride, and the
     *      offsets publish once every row is done.
     *  @param[out] scored The pairs scored per row across every shard, above the stride when
     *      truncated; under a scan, the pairs kept.
     *  @param[in] config The per-shard walk configuration, applied uniformly, its breadth ignored
     *      under a scan.
     *  @param[in,out] executor The threads; one row occupies one at a time.
     *  @param[in,out] traversals One context per executor thread, as the batched @ref search takes.
     *
     *  A full row keeps an arbitrary subset of its pairs under a walk. A thread that fails stops
     *  its own row and leaves the rest to finish; the first failure is what the merged result
     *  reports, and the tape stays unpublished. Refused with @c config_mismatch_k unless the shards
     *  walk by dot.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k The shards do not walk by dot.
     *  - @c unsupported_k @p config names neither a walk nor a scan.
     *  - @c uninitialized_k No shard is placed, or @p executor holds no pool
     *    or no workers.
     *  - @c config_mismatch_k @p scored or the tape disagree with the batch, @p traversals holds
     *    other than one context per thread, or the rows' width differs from the dimensions.
     *  - @c contended_k Another call holds @p executor, its pool runs another dispatch, or two
     *    destructive windows overlapped a row.
     *  - @c overflow_k A queue capacity wraps when rounded.
     */
    template <typename product_type_>
    expected<search_result_t> multiply(rows_view_t left, products_tape<key_t, product_type_> products,
                                       std::span<std::size_t> scored, search_config_t config, executor_t& executor,
                                       std::span<traversal_context_t> traversals) noexcept {
        return multiply_if(left, products, scored, config, dummy_predicate_t {}, executor, traversals);
    }

    /**
     *  @brief The filtered sibling of @ref multiply, sharing its whole contract: only pairs whose
     *      key @p predicate admits are written and counted, while the walks route through the rest.
     *
     *  @param[in] left The left-hand rows, as @ref multiply takes them.
     *  @param[out] products The tape of kept pairs, as @ref multiply writes it.
     *  @param[out] scored The pairs scored per row, as @ref multiply counts them.
     *  @param[in] config The per-shard walk configuration, applied uniformly.
     *  @param[in] predicate The per-key admission predicate, asked about a scored quad of keys at a
     *      time; a scan asks it only about rows nearer than a row's worst kept pair.
     *  @param[in,out] executor The threads, as @ref multiply takes them.
     *  @param[in,out] traversals One context per executor thread, as the batched @ref search takes.
     */
    template <typename product_type_, typename predicate_type_>
    expected<search_result_t> multiply_if(rows_view_t left, products_tape<key_t, product_type_> products,
                                          std::span<std::size_t> scored, search_config_t config,
                                          predicate_type_&& predicate, executor_t& executor,
                                          std::span<traversal_context_t> traversals) noexcept {
        using fold_t =
            multiply_fold<key_t, product_type_, keyed_ids<keyed_t>, std::remove_reference_t<predicate_type_>&>;
        std::size_t const rows_count = left.rows, stride = products.stride;
        if (kernel() != nk_kernel_dot_k) return {{}, status_t::config_mismatch_k};
        if (shards_.empty()) return {{}, status_t::uninitialized_k};
        if (status_t const refusal = products.refusal(rows_count, scored); failed(refusal)) return {{}, refusal};
        if (!rows_count) return {{}, status_t::success_k};
        auto multiply_rows = [&]() noexcept -> expected<search_result_t> {
            if (config.traversal == traversal_t::scan_k) {
                std::memset(scored.data(), 0, scored.size_bytes());
                if (!stride) return {{}, status_t::success_k};
                return scan_rows_(left, executor, traversals, config, products_sink<product_type_> {products, scored},
                                  predicate);
            }
            auto multiply_query = [&](std::size_t row, std::size_t, traversal_context_t& traversal,
                                      std::span<node_id_t const> watermarks, u64_t) noexcept {
                vector_view_t const left_row =
                    shards_[0].index->import_rows_(left, row, 1, traversal).row(0).as_vector();
                fold_t fold {products.keys.subspan(products.staging_of(row), stride),
                             products.scores.subspan(products.staging_of(row), stride),
                             {},
                             predicate};
                status_t refusal = status_t::success_k;
                for (std::size_t shard = 0; shard != watermarks.size(); ++shard) {
                    fold.resolve.keys = &shards_[shard].index->keys();
                    expected<search_result_t> const walked = shards_[shard].index->walk_(
                        left_row, fold, dummy_rescoring_t {}, traversal, config, watermarks[shard]);
                    if (!walked) {
                        refusal = walked.status;
                        break;
                    }
                }
                scored[row] = fold.scored;
                return refusal;
            };
            return for_each_row_(rows_count, executor, traversals, config,
                                 staged_rows_bytes(vector_dtype_k, 1, left.columns), multiply_query, scored);
        };
        expected<search_result_t> const multiplied = multiply_rows();
        if (multiplied) products.publish_runs(scored);
        return multiplied;
    }

    /** NumKong's aggregation over the stored scalar's dots, what a MaxSim total sums in. */
    using total_t = typename scalar_t::reduce_moments_sum_t;

    /**
     *  @brief Late-interaction scores of query groups against every shard: each query row takes its
     *      best dot within a key, and those maxima sum to that key's score, exact for every key the
     *      call keeps; a walk never folds the keys its own dots bound below the kept ones.
     *
     *  @param[in] queries The token rows of every query, laid end to end.
     *  @param[in] query_offsets Where every query's rows start in @p queries, then where the last
     *      ends: one more than the queries.
     *  @param[out] scored Keys each query scored, before the wanted count cut any, one per query.
     *  @param[out] matches The tape each query's kept keys land on in rank order, largest total
     *      first, from its offset, their totals as @p score_type_; the tape's stride bounds the
     *      ranking of every query.
     *  @param[in] config The per-shard walk configuration, applied uniformly: each row nominates as
     *      many keys as its expansion factor, a @c traversal_t::scan_k scan nominates every key,
     *      and each query keeps its best keys, zero keeping every key it scored.
     *  @param[in,out] executor The threads; one query occupies one at a time, end to end.
     *  @param[in,out] traversals One context per executor thread, as the batched @ref search takes;
     *      a query keeps its rows, maxima, ranking and candidates in its thread's staged queries.
     *
     *  Refused with @c shape_mismatch_k for a query without rows, and with @c config_mismatch_k
     *  for offsets past the rows, a @p scored or tape of another length, a ranking wider than the
     *  tape's stride, or unless the shards walk by dot.
     *
     *  @par Statuses:
     *  - @c uninitialized_k No shard is placed, or @p executor holds no pool
     *    or no workers.
     *  - @c config_mismatch_k The shards do not walk by dot, @p query_offsets is empty or passes
     *    the rows, @p scored or the tape differs in length from the queries, or a query keeps more
     *    than the tape's stride.
     *  - @c shape_mismatch_k A query holds no rows.
     *  - @c config_mismatch_k @p traversals holds other than one context per thread, or the rows'
     *    width differs from the dimensions.
     *  - @c contended_k Another call holds @p executor, its pool runs another dispatch, or a
     *    destructive window overlapped a block.
     *  - @c overflow_k A queue capacity wraps when rounded.
     */
    template <typename score_type_>
    expected<search_result_t> maxsim(rows_view_t queries, std::span<std::size_t const> query_offsets,
                                     std::span<std::size_t> scored, products_tape<key_t, score_type_> matches,
                                     maxsim_config_t config, executor_t& executor,
                                     std::span<traversal_context_t> traversals) noexcept {
        return maxsim_if(queries, query_offsets, scored, matches, config, dummy_predicate_t {}, executor, traversals);
    }

    /**
     *  @brief The filtered sibling of @ref maxsim, sharing its whole contract: a key @p predicate
     *      rejects is never nominated, so it adds to no score.
     *
     *  @param[in] queries The token rows of every query, laid end to end.
     *  @param[in] query_offsets Where every query's rows start in @p queries, then where the last
     *      one ends.
     *  @param[out] scored Keys each query scored, before the wanted count cut any, one per query.
     *  @param[out] matches The tape of kept keys, as @ref maxsim writes it.
     *  @param[in] maxsim_config The per-shard walk configuration, applied uniformly, and the keys
     *      each query keeps, the best ones, where zero keeps every key it scored.
     *  @param[in] predicate The per-key admission predicate: a walk asks it as the predicated
     *      @ref search_if asks it, a scan once per key and block of queries.
     *  @param[in,out] executor The threads, as @ref maxsim takes them.
     *  @param[in,out] traversals One context per executor thread, as the batched @ref search takes.
     *
     *  @par Statuses:
     *  - @c uninitialized_k No shard is placed, or @p executor holds no pool
     *    or no workers.
     *  - @c config_mismatch_k The shards do not walk by dot, @p query_offsets is empty or passes
     *    the rows, @p scored or the tape differs in length from the queries, or a query keeps more
     *    than the tape's stride.
     *  - @c shape_mismatch_k A query holds no rows.
     *  - @c config_mismatch_k @p traversals holds other than one context per thread, or the rows'
     *    width differs from the dimensions.
     *  - @c contended_k Another call holds @p executor, its pool runs another dispatch, or a
     *    destructive window overlapped a block.
     *  - @c overflow_k A queue capacity wraps when rounded.
     */
    template <typename score_type_, typename predicate_type_>
    expected<search_result_t> maxsim_if(rows_view_t queries, std::span<std::size_t const> query_offsets,
                                        std::span<std::size_t> scored, products_tape<key_t, score_type_> matches,
                                        maxsim_config_t maxsim_config, predicate_type_&& predicate,
                                        executor_t& executor, std::span<traversal_context_t> traversals) noexcept {
        static_assert((!aggregating_predicate<std::remove_cvref_t<predicate_type_>, key_t, walk_payload_t> &&
                       !aggregating_predicate<std::remove_cvref_t<predicate_type_>, key_t, scan_payload_t>) ||
                          per_thread_predicates<std::remove_cvref_t<predicate_type_>>,
                      "An engine driving its own threads takes an aggregating predicate per thread");
        if (shards_.empty()) return {{}, status_t::uninitialized_k};
        if (kernel() != nk_kernel_dot_k || query_offsets.empty()) return {{}, status_t::config_mismatch_k};
        std::size_t const queries_count = query_offsets.size() - 1;
        for (std::size_t query = 0; query != queries_count; ++query)
            if (query_offsets[query + 1] <= query_offsets[query]) return {{}, status_t::shape_mismatch_k};
        if (query_offsets.back() > queries.rows) return {{}, status_t::config_mismatch_k};
        if (status_t const refusal = matches.refusal(queries_count, scored); failed(refusal)) return {{}, refusal};
        if (!queries_count) return {{}, status_t::success_k};

        search_config_t config = maxsim_config.search;
        std::size_t const wanted = maxsim_config.wanted, stride = matches.stride;
        // Every query walks its rows on one thread, so the driver sizes walks whatever nominates.
        traversal_t const nominating = config.traversal;
        config.traversal = traversal_t::walk_k;
        config.expansion_factor = static_cast<typename index_t::candidates_count_t>(
            shards_[0].index->non_zero_expansion_factor(config));
        // Without a bound, a query keeps every candidate it can have: its walks', or every row.
        std::size_t const visible = size();
        auto capacity_of = [&](std::size_t rows) noexcept {
            return wanted ? wanted : nominating == traversal_t::walk_k ? rows * config.expansion_factor : visible;
        };
        // A scan with a matrix engine reads the shards once per block of queries, a block a thread.
        std::size_t const threads = max_of<std::size_t>(traversals.size(), 1);
        std::size_t const block_queries = index_t::has_tiles_k && nominating == traversal_t::scan_k
                                              ? divide_round_up(queries_count, threads)
                                              : 1;
        std::size_t staged_bytes = 0;
        for (std::size_t first = 0; first < queries_count; first += block_queries) {
            std::size_t const taking = min_of(block_queries, queries_count - first);
            std::size_t const rows = query_offsets[first + taking] - query_offsets[first];
            if (capacity_of(rows) > stride) return {{}, status_t::config_mismatch_k};
            staged_bytes = max_of(staged_bytes, maxsim_staged_bytes_(queries.columns, rows, taking, nominating, config,
                                                                     capacity_of(rows)));
        }
        auto maxsim_block = [&](std::size_t block, std::size_t thread, traversal_context_t& traversal,
                                std::span<node_id_t const> watermarks, u64_t epoch) noexcept {
            std::size_t const first = block * block_queries, taking = min_of(block_queries, queries_count - first);
            std::size_t const rows = query_offsets[first + taking] - query_offsets[first];
            return maxsim_block_(queries, query_offsets.subspan(first, taking + 1), first, nominating, config,
                                 capacity_of(rows), scored.subspan(first, taking), traversal, watermarks, epoch,
                                 thread_predicate(predicate, thread), matches);
        };
        std::size_t const candidates = max_of(index_t::round_capacity(connectivity()),
                                              index_t::chunk_schedule_t::nodes_in(0));
        expected<search_result_t> const ranked = for_each_row_(divide_round_up(queries_count, block_queries), executor,
                                                               traversals, config, staged_bytes, maxsim_block, {},
                                                               predicate, candidates);
        if (ranked) matches.publish_runs(scored);
        return ranked;
    }

    /**
     *  @brief Softmax attention of query heads over every key their walks score across the shards.
     *
     *  Each token walks every shard in turn, folding into one running state per head, since the
     *  shards are disjoint and online softmax is order-free, and its rows normalize once after the
     *  last shard.
     *
     *  @param[in] queries Row t · @p head_count + h is head h of token t; one walk per token and
     *      shard. Any scalar NumKong casts into the stored one; each thread imports its own
     *      token's heads.
     *  @param[in] head_count Query heads sharing this key-value head; it must divide the rows.
     *  @param[in] value_lookup Resolves a scored batch's keys to their value rows, as wide as the keys and stored
     *      like them: <tt>void(std::span&lt;key_t const&gt;, std::span&lt;vector_view_t&gt;)</tt>,
     *      one row per key. Keys come from each shard's keys, so only scored nodes resolve;
     *      one without a row refuses with @c shape_mismatch_k.
     *  @param[in] score_scale The factor turning a dot into a logit.
     *  @param[out] outputs The attended rows, one per query row, as wide as the keys.
     *      Packed @c f32 rows accumulate in place, any others in the thread's staged queries
     *      until their token exports them.
     *  @param[out] carry Every query row's softmax state, which merges this call with other
     *      branches, or empty to keep none.
     *  @param[in] config The per-shard walk configuration, applied uniformly.
     *  @param[in,out] executor The threads; one token occupies one at a time.
     *  @param[in,out] traversals One context per executor thread, as the batched @ref search takes.
     *  @return The query rows attended in @c count.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k The shards do not walk by dot.
     */
    template <typename value_lookup_type_>
    expected<search_result_t> attend(rows_view_t queries, std::size_t head_count, value_lookup_type_&& value_lookup,
                                     f32_t score_scale, rows_span_t outputs, std::span<softmax_state_t> carry,
                                     search_config_t config, executor_t& executor,
                                     std::span<traversal_context_t> traversals) noexcept {
        return attend_if(queries, head_count, value_lookup, score_scale, outputs, carry, config, dummy_predicate_t {},
                         executor, traversals);
    }

    /**
     *  @brief The filtered sibling of @ref attend, sharing its whole contract: a key @p predicate
     *      rejects still routes the walks, but folds into no head and resolves no value row.
     *
     *  @param[in] queries Row t · @p head_count + h is head h of token t.
     *  @param[in] head_count Query heads sharing this key-value head; it must divide the rows.
     *  @param[in] value_lookup Resolves a scored batch's keys to their value rows, as @ref attend
     *      takes it.
     *  @param[in] score_scale The factor turning a dot into a logit.
     *  @param[out] outputs The attended rows, one per query row.
     *  @param[out] carry Every query row's softmax state.
     *  @param[in] config The per-shard walk configuration, applied uniformly.
     *  @param[in] predicate The per-key admission predicate, asked about a scored quad of keys at a
     *      time; a scan asks it about every row.
     *  @param[in,out] executor The threads, as @ref attend takes them.
     *  @param[in,out] traversals One context per executor thread, as the batched @ref search takes.
     *
     *  A @c traversal_t::scan_k scan folds every published row into every query row's softmax, so
     *  each query row attends exactly, as a head of its own.
     */
    template <typename value_lookup_type_, typename predicate_type_>
    expected<search_result_t> attend_if(rows_view_t queries, std::size_t head_count, value_lookup_type_&& value_lookup,
                                        f32_t score_scale, rows_span_t outputs, std::span<softmax_state_t> carry,
                                        search_config_t config, predicate_type_&& predicate, executor_t& executor,
                                        std::span<traversal_context_t> traversals) noexcept {
        using admits_t = node_predicate<std::remove_reference_t<predicate_type_>, keyed_ids<keyed_t>>;
        using lookup_t = shard_value_lookup<value_lookup_type_>;
        std::size_t const rows_count = queries.rows, columns = queries.columns;
        if (kernel() != nk_kernel_dot_k) return {{}, status_t::config_mismatch_k};
        if (shards_.empty()) return {{}, status_t::uninitialized_k};
        if (!head_count || rows_count % head_count) return {{}, status_t::shape_mismatch_k};
        if (outputs.rows != rows_count || outputs.columns != columns || (!carry.empty() && carry.size() != rows_count))
            return {{}, status_t::config_mismatch_k};
        if (!rows_count) return {{}, status_t::success_k};

        // Past the token's imported heads, their `f32` rows, then their softmax states.
        using slice_t = arena_allocator<byte_t>;
        bool const in_place = outputs.template holds<f32_t>() && outputs.stride == columns * sizeof(f32_t);
        std::size_t const heads_bytes = staged_rows_bytes(vector_dtype_k, head_count, columns);
        std::size_t const accumulators_bytes = slice_t::reserved_bytes(head_count * columns * sizeof(f32_t));
        auto attend_token = [&](std::size_t token, std::size_t, traversal_context_t& traversal,
                                std::span<node_id_t const> watermarks, u64_t) noexcept {
            std::size_t const first = token * head_count;
            byte_t* const accumulators = traversal.staged_queries.data() + heads_bytes;
            byte_t* const states = accumulators + accumulators_bytes;
            status_t status = status_t::success_k;
            lookup_t shard_lookup {nullptr, value_lookup};
            admits_t admits {predicate, {}};
            attention_rescoring<quads_kit_t, elementwise_kit_t, lookup_t, admits_t&> const attention {
                shards_[0].index->import_rows_(queries, first, head_count, traversal),
                shard_lookup,
                score_scale,
                {reinterpret_cast<f32_t*>(in_place ? outputs.row(first) : accumulators), head_count, columns},
                carry.empty() ? std::span {reinterpret_cast<softmax_state_t*>(states), head_count}
                              : carry.subspan(first, head_count),
                status,
                admits};
            attention.clear();
            for (std::size_t shard = 0; shard != watermarks.size(); ++shard) {
                shard_lookup.keys = admits.key_of.keys = &shards_[shard].index->keys();
                expected<search_result_t> const walked = shards_[shard].index->walk_(
                    attention.queries.row(0).as_vector(), dummy_fold_t {}, attention, traversal, config,
                    watermarks[shard]);
                if (!walked) return walked.status;
                if (failed(status)) return status;
            }
            attention.finish();
            if (!in_place)
                for (std::size_t head = 0; head != head_count; ++head)
                    export_row<elementwise_kit_t>(attention.outputs.row(head).data(), nk_f32_k, outputs, first + head);
            return status_t::success_k;
        };
        std::size_t const staged_bytes = heads_bytes + accumulators_bytes + head_count * sizeof(softmax_state_t);
        expected<search_result_t> attended =
            config.traversal == traversal_t::scan_k
                ? scan_rows_(queries, executor, traversals, config,
                             attention_sink<value_lookup_type_> {value_lookup, score_scale, outputs, carry}, predicate)
                : for_each_row_(rows_count / head_count, executor, traversals, config, staged_bytes, attend_token, {});
        if (attended) attended.value.count = rows_count;
        return attended;
    }

#pragma endregion Search

#pragma region Checkpoints

    /** What a host checkpoint does beyond the shared plans: every shard opens its own plan on the
     *  calling thread, and a save settles instead of closing the dispensers, so nothing else ends.
     *  A load closes the index under its gate, and opens shard generations past the placed shards
     *  where a checkpoint's parts outnumber them. */
    struct checkpoint_residency_t {

        /** The index a load closes and builds new shard generations' shards for; a save names
         *  none. */
        ram_indexes* owner = nullptr;

        /**
         *  Closes every shard of the loading index under its unique side.
         *
         *  @par Statuses:
         *  - @c contended_k A call holds the gate, or a batch or plan holds a shard's dispenser.
         */
        status_t close(shards_t&) noexcept { return owner->close_gated_(); }

        /**
         *  Leaves part @p part the live shard it names, a placed one, or opens shard generations on
         *  the placed shards' memory domains until it exists.
         *
         *  @par Statuses:
         *  - @c contended_k The set is not sealed while a shard generation must open.
         */
        status_t open_shard_for_part(shards_t& shards, std::size_t part) noexcept {
            while (shards.size() <= shards.retired_shards() + part)
                if (status_t const opened = shards.open_shard_generation_past_seal_(
                        shards.open_shard_generation(),
                        [this](shard_t const& like) noexcept { return owner->build_shard_(like); });
                    failed(opened))
                    return opened;
            return status_t::success_k;
        }

        /**
         *  Opens the save plan of shard @p shard into @p columns, the shard's own @c plan_save.
         *
         *  @par Statuses:
         *  - @c contended_k A dispensed batch has yet to publish.
         *  - @c config_mismatch_k A column is too short for the rows, or norms are given to a shard
         *    that stores none.
         */
        static expected<typename index_t::save_plan_t> open_shard_save(shards_t& shards, std::size_t shard,
                                                                       typename index_t::columns_t const& columns,
                                                                       synchronization_t synchronization) noexcept {
            return shards[shard].index->plan_save(columns, synchronization);
        }

        /**
         *  @brief Opens the load plan of @p rows in shard @p shard, the shard's own @c plan_load.
         *
         *  @par Statuses:
         *  - @c contended_k A dispensed batch has yet to publish, or another claim won first.
         *  - @c overflow_k The rows pass the ID space, the reserved ceiling or the chunk directory.
         */
        static expected<typename index_t::load_plan_t> open_shard_load(shards_t& shards, std::size_t shard,
                                                                       std::size_t rows) noexcept {
            return shards[shard].index->plan_load(rows);
        }
        static void end_save(shards_t&) noexcept {}
    };

    /** One save in flight across every shard: one shard plan each, the parts they write into,
     *  and the tail @c finish commits - the keys, the headers, and the manifest. */
    using save_plan_t = shard_save_plan<shards_t, checkpoint_residency_t>;

    /** One load in flight across every shard: the mapped parts, one shard plan each, and the flat
     *  steps their rows arrive in - one per part's share of one storage chunk. */
    using load_plan_t = shard_load_plan<shards_t, checkpoint_residency_t>;

    /**
     *  @brief Bytes a buffer @ref save under @p config writes: a lone live shard's monolith, or the
     *      manifest and every live shard's part it lists; zero when the config is invalid or the
     *      index is unplaced.
     *
     *  The row counts are read now, so a buffer sized here fits a save no row enters first.
     *
     *  @par Statuses:
     *  - @c uninitialized_k No shard is placed.
     *  - @c config_mismatch_k @p config does not validate.
     */
    expected<std::size_t> storage_requirements(save_config_t const& config) const noexcept {
        return shard_storage_requirements(shards_, config);
    }

    /**
     *  @brief Opens a save under @p path: a lone live shard's monolith at the path itself, or one
     *      monolith per live shard beside the manifest the path names.
     *
     *  Every part is framed, sized and mapped here, and every header is written in @c finish, so a
     *  failed or cancelled save reads as absent.
     *
     *  @param[in] path The monolith's path, or the manifest's - its parts go beside it as
     *      @c -00001-of-0000S.safegraphs after the stem.
     *  @param[in] config The alignment, synchronization contract and provenance.
     *
     *  Under @c concurrent_k contention is detected, not excluded: a row inserted mid-save
     *  reports @c contended_k. A settled save holds every batch on all of its shards or on none:
     *  a batch dispenses on every shard before it publishes on any, so a shard still awaiting its
     *  publication refuses its plan, and every shard's plan opens here, before any @c finish
     *  re-checks.
     *
     *  @par Statuses:
     *  - @c uninitialized_k No shard is placed.
     *  - @c unknown_kind_k The stored scalar has no interchange dtype.
     *  - @c config_mismatch_k @p config does not validate.
     *  - @c invalid_argument_k A part path does not parse.
     *  - @c overflow_k A device ends before a part does.
     *  - @c contended_k A dispensed batch has yet to publish.
     */
    expected<save_plan_t> plan_save(std::string_view path, save_config_t const& config) noexcept {
        return save_plan_t::open(shards_, {}, path, config);
    }

    /** Opens a save into @p buffer - the in-memory sibling of the path form: a lone live shard's
     *  monolith, or the manifest leading the buffer and the parts it lists. Size the bytes by
     *  @ref storage_requirements; fewer answer @c overflow_k.
     *
     *  @par Statuses:
     *  - @c uninitialized_k No shard is placed.
     *  - @c unknown_kind_k The stored scalar has no interchange dtype.
     *  - @c config_mismatch_k @p config does not validate.
     *  - @c overflow_k @p buffer is shorter than the checkpoint.
     *  - @c contended_k A dispensed batch has yet to publish.
     */
    expected<save_plan_t> plan_save(bytes_span_t buffer, save_config_t const& config) noexcept {
        return save_plan_t::open(shards_, {}, buffer, config);
    }

    /**
     *  @brief Opens a load of the checkpoint at @p path: the content says monolith or manifest,
     *      every part's header is vetted before a byte is copied, the reservation comes from
     *      @p config, and part @c p goes to live shard @c p, opened past the placed ones where
     *      the parts outnumber them.
     *
     *  @param[in] path A monolith, or the manifest whose parts sit beside it.
     *  @param[in] config The rows to reserve; zero fits the checkpoint.
     *
     *  The index must be empty; the plan reports the rows it will load through @c rows().
     *
     *  @par Statuses:
     *  - @c uninitialized_k No shard is placed.
     *  - @c contended_k A call holds the gate, a batch or plan holds a shard's dispenser, or a
     *    dispensed batch has yet to publish.
     *  - @c config_mismatch_k The index holds entries, or the checkpoint's geometry, dtypes or
     *    metric differ from the shards'.
     *  - @c invalid_argument_k The checkpoint or a part path does not parse.
     *  - @c malformed_k The checkpoint lists no part, a part or its header does not parse, or its
     *    connectivity or dimensions are unusable.
     *  - @c unknown_kind_k A shard reserving for the checkpoint cannot finalize its kernel.
     *  - @c overflow_k A part passes a shard's bound.
     */
    expected<load_plan_t> plan_load(std::string_view path, load_config_t const& config) noexcept {
        return load_plan_t::open(shards_, {this}, path, config);
    }

    /**
     *  Opens a load of the checkpoint held in @p buffer: a lone monolith, or the manifest leading
     *  the buffer and the parts it lists.
     *
     *  @par Statuses:
     *  - @c uninitialized_k No shard is placed.
     *  - @c contended_k A call holds the gate, a batch or plan holds a shard's dispenser, or a
     *    dispensed batch has yet to publish.
     *  - @c config_mismatch_k The index holds entries, or the checkpoint's geometry, dtypes or
     *    metric differ from the shards'.
     *  - @c malformed_k The manifest lists no part, a part or its header does not parse, or its
     *    connectivity or dimensions are unusable.
     *  - @c unknown_kind_k A shard reserving for the checkpoint cannot finalize its kernel.
     *  - @c overflow_k A part passes a shard's bound.
     */
    expected<load_plan_t> plan_load(bytes_view_t buffer, load_config_t const& config) noexcept {
        return load_plan_t::open(shards_, {this}, buffer, config);
    }

    /**
     *  @brief Serializes the index under @p path - the one-shot over @ref plan_save, its steps
     *      fanned out over @p executor.
     *
     *  @param[in] path The monolith's path, or the manifest's.
     *  @param[in] config The alignment, synchronization contract and provenance.
     *  @param[in,out] executor The threads the shard writes fan out over.
     *  @param[in] on_saved Fires per finished shard with its index; answering @c cancel_k stops
     *      the remaining shards and reports @c cancelled_k.
     *
     *  @par Statuses:
     *  - @c uninitialized_k No shard is placed, or @p executor holds no
     *    pool, arenas or workers.
     *  - @c unknown_kind_k The stored scalar has no interchange dtype.
     *  - @c config_mismatch_k @p config does not validate, or a header does not fit before its
     *    data.
     *  - @c invalid_argument_k A part or manifest path does not parse.
     *  - @c overflow_k A device ends before a part or the manifest does.
     *  - @c contended_k A dispensed batch has yet to publish, a counter moved under
     *    @c concurrent_k, another call holds @p executor, or its pool runs another dispatch.
     */
    template <typename progress_type_ = dummy_progress_t>
    status_t save(std::string_view path, save_config_t const& config, executor_t& executor,
                  progress_type_&& on_saved = {}) noexcept {
        auto [plan, status] = plan_save(path, config);
        if (failed(status)) return status;
        if (status_t const drained = drain_steps_(plan, executor); failed(drained)) return drained;
        return plan.finish(on_saved);
    }

    /**
     *  Serializes the index into @p buffer - the one-shot over the bytes form of
     *  @ref plan_save.
     *
     *  @par Statuses:
     *  - @c uninitialized_k No shard is placed, or @p executor holds no
     *    pool, arenas or workers.
     *  - @c unknown_kind_k The stored scalar has no interchange dtype.
     *  - @c config_mismatch_k @p config does not validate, or a header does not fit before its
     *    data.
     *  - @c overflow_k @p buffer is shorter than the checkpoint.
     *  - @c contended_k A dispensed batch has yet to publish, a counter moved under
     *    @c concurrent_k, another call holds @p executor, or its pool runs another dispatch.
     */
    template <typename progress_type_ = dummy_progress_t>
    status_t save(bytes_span_t buffer, save_config_t const& config, executor_t& executor,
                  progress_type_&& on_saved = {}) noexcept {
        auto [plan, status] = plan_save(buffer, config);
        if (failed(status)) return status;
        if (status_t const drained = drain_steps_(plan, executor); failed(drained)) return drained;
        return plan.finish(on_saved);
    }

    /**
     *  @brief Restores the checkpoint at @p path into the placed shards - the one-shot over
     *      @ref plan_load, its steps fanned out over @p executor.
     *
     *  The index must be empty. A cancelled or failed load clears it and reports why, so nothing
     *  is published and the next load starts clean.
     *
     *  @par Statuses:
     *  - @c uninitialized_k No shard is placed, or @p executor holds no pool, arenas or workers.
     *  - @c contended_k A call holds the gate, a batch or plan holds a shard's dispenser, a
     *    dispensed batch has yet to publish, another call holds @p executor, or its pool runs
     *    another dispatch.
     *  - @c config_mismatch_k The index holds entries, the checkpoint's geometry, dtypes or metric
     *    differ from the shards', or a step's rows rebase past its plan.
     *  - @c invalid_argument_k The checkpoint or a part path does not parse.
     *  - @c malformed_k The checkpoint lists no part, a part, its header or its key column does
     *    not parse, or its connectivity or dimensions are unusable.
     *  - @c unknown_kind_k A shard reserving for the checkpoint cannot finalize its kernel.
     *  - @c overflow_k A part passes a shard's bound.
     */
    template <typename progress_type_ = dummy_progress_t>
    status_t load(std::string_view path, load_config_t const& config, executor_t& executor,
                  progress_type_&& on_loaded = {}) noexcept {
        status_t settled = status_t::success_k;
        {
            auto [plan, status] = plan_load(path, config);
            if (failed(status)) return status; // Nothing was claimed, so nothing needs undoing.
            settled = drain_steps_(plan, executor);
            if (succeeded(settled)) settled = plan.finish(on_loaded);
        }
        // Nothing is published by a failed or cancelled load: the shards go back to empty, once
        // the plan has died and released whatever it claimed.
        if (failed(settled)) [[maybe_unused]]
            status_t const cleared = clear();
        return settled;
    }

    /**
     *  Restores the checkpoint held in @p buffer - the one-shot over the bytes form of
     *  @ref plan_load.
     *
     *  @par Statuses:
     *  - @c uninitialized_k No shard is placed, or @p executor holds no pool, arenas or workers.
     *  - @c contended_k A call holds the gate, a batch or plan holds a shard's dispenser, a
     *    dispensed batch has yet to publish, another call holds @p executor, or its pool runs
     *    another dispatch.
     *  - @c config_mismatch_k The index holds entries, the checkpoint's geometry, dtypes or metric
     *    differ from the shards', or a step's rows rebase past its plan.
     *  - @c malformed_k The manifest lists no part, a part, its header or its key column does not
     *    parse, or its connectivity or dimensions are unusable.
     *  - @c unknown_kind_k A shard reserving for the checkpoint cannot finalize its kernel.
     *  - @c overflow_k A part passes a shard's bound.
     */
    template <typename progress_type_ = dummy_progress_t>
    status_t load(bytes_view_t buffer, load_config_t const& config, executor_t& executor,
                  progress_type_&& on_loaded = {}) noexcept {
        status_t settled = status_t::success_k;
        {
            auto [plan, status] = plan_load(buffer, config);
            if (failed(status)) return status; // Nothing was claimed, so nothing needs undoing.
            settled = drain_steps_(plan, executor);
            if (succeeded(settled)) settled = plan.finish(on_loaded);
        }
        if (failed(settled)) [[maybe_unused]]
            status_t const cleared = clear();
        return settled;
    }

    /**
     *  The checkpoint's metadata without loading it - @ref safegraphs_read_metadata.
     *
     *  @par Statuses:
     *  - @c invalid_argument_k A path does not parse.
     *  - @c malformed_k The file is empty, or its manifest or header does not parse.
     */
    static expected<safegraphs_metadata_t> metadata(std::string_view path) noexcept {
        return safegraphs_read_metadata(path);
    }

    /**
     *  The metadata of one monolith held in memory.
     *
     *  @par Statuses:
     *  - @c malformed_k The manifest or the header does not parse.
     */
    static expected<safegraphs_metadata_t> metadata(bytes_view_t buffer) noexcept {
        return safegraphs_read_metadata(buffer);
    }

#pragma endregion Checkpoints

    /** Entries stored under @p key, every shard probed at one instant: the epoch covers the windows
     *  that move or forget entries, and the commit sequence a batch spanning shards, so the count
     *  is exact. */
    std::size_t count(key_t const& key) const noexcept { return shards_.count(key); }

    /** Whether at least one entry is stored under @p key, every shard probed as @ref count does. */
    bool contains(key_t const& key) const noexcept { return shards_.contains(key); }

    /**
     *  @brief Exports up to @p wanted rows stored under @p key into @p vectors in their scalar, and
     *      reports the @b true total under that key - a return above @p wanted means the copy
     *      truncated, so the caller resizes and asks again; zero means absent.
     *
     *  Rows at least as wide as the index take its columns and keep their tails. Safe against
     *  concurrent insertions, which only ever add. The executor rides in the signature unused,
     *  since one key's rows copy on the calling thread, so a caller drives a host index exactly as
     *  it drives a device one.
     */
    std::size_t get(key_t const& key, rows_span_t vectors, std::size_t wanted, executor_t&) const noexcept {
        return shards_.read_until_settled([&]() noexcept { return get_(key, vectors, wanted); });
    }

    /**
     *  @brief The batched @c count: the entries stored under every key of @p keys, one key per
     *      task over the executor's threads.
     *
     *  @param[in] keys The keys to count, repeats and absent ones included.
     *  @param[out] entries The entries under each key, parallel to @p keys; zero means absent.
     *  @param[in,out] executor The threads the lookups fan out over.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k @p entries differs in length from @p keys.
     *  - @c contended_k Another call holds @p executor, or its pool runs another dispatch.
     *  - @c uninitialized_k @p executor holds no pool, or the pool has no workers.
     */
    status_t count(std::span<key_t const> keys, std::span<std::size_t> entries, executor_t& executor) const noexcept {
        if (entries.size() != keys.size()) return status_t::config_mismatch_k;
        auto [executor_claim, claim_status] = executor.claim();
        if (failed(claim_status)) return claim_status;
        return executor.for_tasks(keys.size(), [&](std::size_t position, std::size_t, std::size_t) noexcept {
            entries[position] = count(keys[position]);
        });
    }

    /**
     *  The batched @c contains: one in @p entries where a key of @p keys holds any entry, zero
     *  where it holds none, one key per task over the executor's threads.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k @p entries differs in length from @p keys.
     *  - @c contended_k Another call holds @p executor, or its pool runs another dispatch.
     *  - @c uninitialized_k @p executor holds no pool, or the pool has no workers.
     */
    status_t contains(std::span<key_t const> keys, std::span<std::size_t> entries,
                      executor_t& executor) const noexcept {
        if (entries.size() != keys.size()) return status_t::config_mismatch_k;
        auto [executor_claim, claim_status] = executor.claim();
        if (failed(claim_status)) return claim_status;
        return executor.for_tasks(keys.size(), [&](std::size_t position, std::size_t, std::size_t) noexcept {
            entries[position] = contains(keys[position]) ? 1 : 0;
        });
    }

    /**
     *  @brief The batched @c get: up to @p wanted rows of every key of @p keys, and every key's
     *      true total, one key per task over the executor's threads.
     *
     *  @param[in] keys The keys to export, repeats and absent ones included.
     *  @param[out] vectors @b [keys·wanted,dimensions] in any scalar NumKong casts into, where row
     *      q · @p wanted + i is the i-th row of key q; refused with @c config_mismatch_k when the
     *      rows are another count or narrower than the index.
     *  @param[out] entries The true total under each key, parallel to @p keys; above @p wanted
     *      when its rows truncated, zero when absent.
     *  @param[in] wanted Rows kept per key.
     *  @param[in,out] executor The threads the exports fan out over.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k @p entries differs in length from @p keys, or @p vectors holds
     *    other than @p wanted rows per key or narrower rows than the dimensions.
     *  - @c contended_k Another call holds @p executor, its pool runs another dispatch, or two
     *    windows overlapped the call.
     *  - @c uninitialized_k @p executor holds no pool, or the pool has no workers.
     */
    status_t get(std::span<key_t const> keys, rows_span_t vectors, std::span<std::size_t> entries, std::size_t wanted,
                 executor_t& executor) const noexcept {
        std::size_t const rows = keys.size() * wanted;
        if (entries.size() != keys.size()) return status_t::config_mismatch_k;
        if (vectors.rows != rows || (rows && vectors.columns < dimensions())) return status_t::config_mismatch_k;
        auto [executor_claim, claim_status] = executor.claim();
        if (failed(claim_status)) return claim_status;
        return shards_.read_retrying_once([&]() noexcept {
            return executor.for_tasks(keys.size(), [&](std::size_t position, std::size_t, std::size_t) noexcept {
                entries[position] = shards_.read_committed(
                    [&]() noexcept { return get_(keys[position], vectors.slice(position * wanted, wanted), wanted); });
            });
        });
    }

    /**
     *  @brief Draws up to @p outputs' length entries around the focus keys, or across the whole
     *      index without any, and places them in three coordinates among the shown points of
     *      @p inputs.
     *
     *  The roots, paths and regions of @c sample_plan, below one snapshot of the watermarks: each
     *  draw names a key and its ordinal among that key's entries as @c get orders them, the shown
     *  point or earlier draw it hangs from, and its coordinates. Without shown points every draw's
     *  coordinates are its vector's projection. Equal seeds, shown points and focus over an
     *  unchanged index draw equally, and a shorter @p outputs draws a prefix of a longer one. The
     *  draws run on the calling thread; the scratch comes from the executor's first arena and is
     *  rewound once afterward.
     *
     *  @par Statuses:
     *  - @c uninitialized_k No shard is placed, or @p executor holds no pool or arenas.
     *  - @c shape_mismatch_k An output's or the shown coordinates' length disagrees.
     *  - @c bad_alloc_k The arena cannot hold the scratch.
     *  - @c contended_k Another call holds @p executor, or two windows overlapped the call.
     */
    expected<sample_result_t> sample(sample_inputs<key_t> inputs, sample_outputs<key_t> outputs, sample_config_t config,
                                     executor_t& executor) noexcept {
        using plan_t = sample_plan<dense_sample_source_t>;
        if (shards_.empty()) return {{}, status_t::uninitialized_k};
        auto [executor_claim, claim_status] = executor.claim();
        if (failed(claim_status)) return {{}, claim_status};
        if (status_t const rewound = executor.reserve(0); failed(rewound)) return {{}, rewound};
        typename executor_t::arena_t& arena = executor.arena(0);
        std::size_t const scratch_bytes = plan_t::scratch_bytes(outputs.keys.size(), inputs.shown.size(), config);
        byte_t* const scratch = arena.allocate(scratch_bytes, default_alignment_k);
        f32_t* const vector = reinterpret_cast<f32_t*>(
            arena.allocate(dimensions() * sizeof(f32_t), default_alignment_k));
        if (!scratch || !vector) return {{}, status_t::bad_alloc_k};
        return shards_.read_retrying_once([&]() noexcept -> expected<sample_result_t> {
            std::span<node_id_t> watermarks;
            do {
                std::size_t const shards = shards_.size();
                byte_t* const slots = arena.allocate(shards * sizeof(node_id_t), alignof(node_id_t));
                if (!slots) return {{}, status_t::bad_alloc_k};
                watermarks = {reinterpret_cast<node_id_t*>(slots), shards};
            } while (failed(shards_.snapshot(watermarks).status));
            dense_sample_source_t const source {*this, watermarks, {vector, dimensions()}};
            plan_t plan {source, inputs, outputs, config, {scratch, scratch_bytes}};
            status_t status = plan.draw();
            if (succeeded(status)) status = plan.assign_coordinates();
            return {plan.result(), status};
        });
    }

  private:

    /** Every shard as SAMPLE reads it, below the watermarks of @c watermarks_snapshot; see
     *  @c sample_source. */
    struct dense_sample_source_t {
        using key_t = key_type_;
        using node_id_t = dense::node_id_t;

        ram_indexes const& indexes;
        std::span<node_id_t const> watermarks_snapshot;

        /** The f32 vector the dense projection reads, as wide as the dimensions. */
        std::span<f32_t> vector;

        std::span<node_id_t const> watermarks() const noexcept { return watermarks_snapshot; }

        key_t key_of(shard_node_t node) const noexcept {
            keyed_t const& keys = indexes.shards_[node.shard].index->keys();
            return keys[node.node];
        }

        template <typename callback_type_>
        void nodes_of(key_t const& key, callback_type_&& callback) const noexcept {
            for (std::size_t shard = 0; shard != watermarks_snapshot.size(); ++shard)
                indexes.shards_[shard].index->keys().find_each(
                    key, watermarks_snapshot[shard],
                    [&](node_id_t node) noexcept { callback(shard_node_t {static_cast<u32_t>(shard), node}); });
        }

        /** Reads the neighbor list as a cursor stages it: slot 0 acquired once, waited out while
         *  locked, and a self-loop meaning no neighbors. */
        template <typename callback_type_>
        void neighbors(shard_node_t node, callback_type_&& callback) const noexcept {
            index_t const& shard_index = *indexes.shards_[node.shard].index;
            typename index_t::node_ids_view_t const row = shard_index.neighbors_for(
                index_t::chunk_schedule_t::locate(node.node));
            node_id_t first = atomic_ref<node_id_t const>(row[0]).load(std::memory_order_acquire);
            while (first == node_id_unknown_k) {
                typename index_t::wait_t {}(row.data(), node_id_unknown_k, typename index_t::candidates_count_t(0),
                                            fu::wait_capped_t {});
                first = atomic_ref<node_id_t const>(row[0]).load(std::memory_order_acquire);
            }
            if (first == node.node) return;
            for (std::size_t slot = 0; slot != row.size(); ++slot) {
                node_id_t const next = slot ? atomic_ref<node_id_t const>(row[slot]).load(std::memory_order_relaxed)
                                            : first;
                if (next == node_id_unknown_k) break;
                if (next < watermarks_snapshot[node.shard]) callback(shard_node_t {node.shard, next});
            }
        }

        void distance(shard_node_t from, std::span<shard_node_t const> nodes,
                      std::span<f32_t> distances) const noexcept {
            index_t const& origin = *indexes.shards_[from.shard].index;
            typename index_t::location_t const location = index_t::chunk_schedule_t::locate(from.node);
            vector_view_t const query = origin.vector_for(location);
            vector_norms_t const query_norms = origin.norm_for(location);
            for (std::size_t first = 0; first < nodes.size(); first += quad_lanes_k) {
                std::size_t const lanes = min_of<std::size_t>(quad_lanes_k, nodes.size() - first);
                quad<vector_view_t> lane_rows;
                quad<vector_norms_t> lane_norms;
                for (std::size_t lane = 0; lane != quad_lanes_k; ++lane) {
                    shard_node_t const target = nodes[first + min_of(lane, lanes - 1)];
                    index_t const& shard_index = *indexes.shards_[target.shard].index;
                    typename index_t::location_t const target_location = index_t::chunk_schedule_t::locate(target.node);
                    lane_rows[lane] = shard_index.vector_for(target_location);
                    lane_norms[lane] = shard_index.norm_for(target_location);
                }
                quad<distance_t> const scored = origin.distances(query, query_norms, lane_rows, lane_norms);
                for (std::size_t lane = 0; lane != lanes; ++lane) distances[first + lane] = scored[lane];
            }
        }

        std::size_t walk(shard_node_t from, shard_node_t toward, std::span<shard_node_t> chain) const noexcept {
            return sample_greedy_walk(*this, from, toward, chain);
        }

        std::span<f32_t const> values_of(shard_node_t node) const noexcept {
            rows_span_t const target {reinterpret_cast<byte_t*>(vector.data()), 1, vector.size(), vector.size_bytes(),
                                      nk_f32_k};
            export_row<elementwise_kit_t>(indexes.shards_[node.shard].index->vector(node.node).byte_data(),
                                          vector_dtype_k, target, 0);
            return vector;
        }
    };

    /** The caller's key-taking value lookup, fed a walk's node IDs through one shard's keys. */
    template <typename value_lookup_type_>
    struct shard_value_lookup {

        /** The keys of the shard being walked, rebound per shard. */
        keyed_t const* keys = nullptr;

        /** The caller's lookup, which takes keys. */
        value_lookup_type_& value_lookup;

        template <std::size_t extent_>
        void operator()(std::span<node_id_t const, extent_> nodes,
                        std::span<vector_view_t, extent_> rows) const noexcept {
            std::array<key_t, extent_> node_keys;
            for (std::size_t lane = 0; lane != extent_; ++lane) node_keys[lane] = (*keys)[nodes[lane]];
            value_lookup(std::span<key_t const, extent_>(node_keys), rows);
        }
    };

    /** Bytes one pack of up to @c pack_columns_k query rows takes, each pack on its own line. */
    std::size_t pack_stride_() const noexcept
        requires index_t::has_tiles_k
    {
        nk_size_t bytes = 0;
        if (tiles_kit_t::pack_size(tiles_kit_t::pack_columns_k, dimensions(), &bytes) != nk_success_k) return 0;
        return round_up_to_multiple<std::size_t>(bytes, default_alignment_k);
    }

    /** Stored rows one gather block holds: as many as share @p cache_bytes with one pack, never
     *  fewer than a kit step, and one step when the cache is unknown. */
    std::size_t block_rows_(std::size_t cache_bytes) const noexcept
        requires index_t::has_tiles_k
    {
        std::size_t const pack_stride = pack_stride_(), row_bytes = packed_bytes(vector_dtype_k, dimensions());
        return max_of((cache_bytes - min_of(cache_bytes, pack_stride)) / row_bytes, tiles_kit_t::stream_rows_k);
    }

    /** Takes @p count elements off the front of @p staged, which then starts on the next line. */
    template <typename element_type_>
    static std::span<element_type_> carve_(bytes_span_t& staged, std::size_t count) noexcept {
        std::size_t const bytes = count * sizeof(element_type_);
        std::span<element_type_> const carved = elements_of<element_type_>(staged.first(bytes));
        staged = staged.subspan(arena_allocator<byte_t>::reserved_bytes(bytes));
        return carved;
    }

    /**
     *  @brief Folds keys through the tiles kit for a block of query rows, every key's visible rows
     *      gathered into a block of stored rows, one segment per key.
     *
     *  The block folds against every pack of the query rows once full, and the @c rank_key each
     *  call passes takes each key with its maxima row, one maximum per query row, once all of its
     *  rows folded; a key with more rows than the block carries its maxima row into the next fold.
     *  Carved from @c staged as @c maxsim_staged_bytes_ counts it.
     */
    struct tiled_fold_t {
        tiled_fold_t(ram_indexes const& owner, vectors_view_t query_rows, bytes_span_t staged, std::size_t cache_bytes,
                     std::span<node_id_t const> watermarks, traversal_context_t& traversal) noexcept
            : owner_(owner), watermarks_(watermarks), traversal_(traversal), columns_(query_rows.extent(0)),
              pack_bytes_(owner.pack_stride_()), block_capacity_(owner.block_rows_(cache_bytes)),
              row_bytes_(packed_bytes(vector_dtype_k, owner.dimensions())) {
            packs_ =
                carve_<byte_t>(staged, divide_round_up(columns_, tiles_kit_t::pack_columns_k) * pack_bytes_).data();
            block_ = carve_<byte_t>(staged, block_capacity_ * row_bytes_).data();
            ends_ = carve_<std::size_t>(staged, block_capacity_ + 1);
            keys_ = carve_<key_t>(staged, block_capacity_ + 1);
            maxima_ = {carve_<dot_result_t>(staged, (block_capacity_ + 1) * columns_).data(), block_capacity_ + 1,
                       columns_};
            for (std::size_t first = 0; first < columns_; first += tiles_kit_t::pack_columns_k) {
                std::size_t const width = min_of(tiles_kit_t::pack_columns_k, columns_ - first);
                vectors_view_t const pack_rows = query_rows(nk::range(first, first + width), nk::slice);
                [[maybe_unused]] nk_status_t const packing = tiles_kit_t::pack(
                    pack_rows.raw_values_data(), width, owner.dimensions(),
                    static_cast<std::size_t>(pack_rows.stride_bytes(0)),
                    packs_ + first / tiles_kit_t::pack_columns_k * pack_bytes_, 0, width, nullptr);
                assert(packing == nk_success_k && "A matrix engine packs any block");
            }
        }

        /** Gathers every visible row of @p key as its own segment, folding the block whenever it
         *  fills; a key without visible rows folds nothing and is never ranked. */
        template <typename rank_key_type_>
        void add(key_t const& key, rank_key_type_& rank_key) noexcept {
            keys_[segments_] = key;
            for (dot_result_t& maximum : to_span(maxima_.row(segments_).as_vector()))
                maximum = dot_result_t::finite_min();
            std::size_t folded = 0;
            // A shard of a shard generation opened past the snapshot holds none of its rows.
            owner_.shards_.find_each(key, watermarks_, [&](index_t const& shard_index, node_id_t node) noexcept {
                if (gathered_ == block_capacity_) carry_(rank_key);
                std::memcpy(block_ + gathered_++ * row_bytes_, shard_index.vector(node).byte_data(), row_bytes_);
                ++folded;
            });
            ends_[segments_] = gathered_;
            if (folded) ++segments_;
        }

        /** Folds and ranks whatever the block still holds. */
        template <typename rank_key_type_>
        void finish(rank_key_type_& rank_key) noexcept {
            if (segments_) fold_(segments_, rank_key);
        }

      private:
        ram_indexes const& owner_;
        std::span<node_id_t const> watermarks_;
        traversal_context_t& traversal_;
        std::size_t columns_, pack_bytes_, block_capacity_, row_bytes_;
        byte_t* packs_ = nullptr;
        byte_t* block_ = nullptr;
        std::span<std::size_t> ends_;
        std::span<key_t> keys_;
        nk::matrix_span<dot_result_t> maxima_;
        std::size_t gathered_ = 0, segments_ = 0;

        /** Folds the block's rows into its first @p segments maxima rows against every pack, then
         *  ranks the closed segments. */
        template <typename rank_key_type_>
        void fold_(std::size_t segments, rank_key_type_& rank_key) noexcept {
            vectors_view_t const rows {reinterpret_cast<scalar_t const*>(block_), gathered_, owner_.dimensions()};
            for (std::size_t first = 0; first < columns_; first += tiles_kit_t::pack_columns_k) {
                std::size_t const width = min_of(tiles_kit_t::pack_columns_k, columns_ - first);
                fold_tiles<tiles_kit_t>(
                    rows, packs_ + first / tiles_kit_t::pack_columns_k * pack_bytes_, ends_.first(segments),
                    maxima_(nk::range(std::size_t {0}, segments), nk::range(first, first + width), nk::slice),
                    traversal_.step_dots.span());
            }
            traversal_.computed_distances += gathered_ * columns_;
            for (std::size_t segment = 0; segment != segments_; ++segment)
                rank_key(keys_[segment], to_span(maxima_.row(segment).as_vector()));
        }

        /** Folds a block that filled inside a key: the closed segments rank, and the open one moves
         *  to the front with its maxima to keep accumulating. */
        template <typename rank_key_type_>
        void carry_(rank_key_type_& rank_key) noexcept {
            ends_[segments_] = gathered_;
            fold_(segments_ + 1, rank_key);
            std::memmove(maxima_.row(0).data(), maxima_.row(segments_).data(), columns_ * sizeof(dot_result_t));
            keys_[0] = keys_[segments_];
            gathered_ = segments_ = 0;
        }
    };

    /** One key a MaxSim walk row kept, grouped by key to bound what the key can total. */
    struct alignas(2 * sizeof(key_t)) maxsim_hit_t {

        /** The key the row's walks ranked. */
        key_t key;

        /** The query row whose walks ranked it, counted within the block. */
        u32_t row;

        /** How far its dot rises above the row's worst kept one, in walk distances. */
        distance_t gain;
    };

    /** Larger totals first, ties by key bytes, so every thread count ranks alike. */
    static bool better_(total_t first_total, key_t const& first_key, total_t second_total,
                        key_t const& second_key) noexcept {
        if (first_total != second_total) return first_total > second_total;
        return std::memcmp(&first_key, &second_key, sizeof(key_t)) < 0;
    }

    /** Keeps @p key at @p total among @p best and its keys, @p kept of them so far: appended while
     *  there is room, else in place of the @p worst slot it beats, which a full ranking keeps
     *  current, so a rejected key costs one comparison. */
    static void rank_into_(std::span<maxsim_ranked<total_t>> best, std::span<key_t> best_keys, std::size_t& kept,
                           std::size_t& worst, key_t const& key, total_t total) noexcept {
        if (best.empty()) return;
        if (kept != best.size()) {
            best[kept] = {static_cast<u32_t>(kept), total};
            best_keys[kept++] = key;
            if (kept != best.size()) return;
        }
        else {
            if (!better_(total, key, best[worst].total, best_keys[best[worst].candidate])) return;
            best[worst].total = total;
            best_keys[best[worst].candidate] = key;
        }
        worst = 0;
        for (std::size_t slot = 1; slot != kept; ++slot)
            if (better_(best[worst].total, best_keys[best[worst].candidate], best[slot].total,
                        best_keys[best[slot].candidate]))
                worst = slot;
    }

    /** Whether visible @p node of shard @p shard leads its @p key under the snapshot @p visible:
     *  no earlier shard holds the key below its watermark, and no lower node of this shard does. A
     *  function of the snapshot alone, so a scan nominates every key exactly once per snapshot. */
    bool leads_(std::size_t shard, node_id_t node, key_t const& key,
                std::span<node_id_t const> visible) const noexcept {
        index_t const& own = *shards_[shard].index;
        bool preceded = false;
        shards_.find_each(key, visible.first(shard + 1), [&](index_t const& shard_index, node_id_t other) noexcept {
            preceded = preceded || &shard_index != &own || other < node;
        });
        return !preceded;
    }

    /** Stages each query's kept keys of @p best on @p matches, largest first, the q-th query at the
     *  stride of query @p first_query + q. */
    template <typename score_type_>
    static void write_ranked_(std::span<maxsim_ranked<total_t>> best, std::span<key_t const> best_keys,
                              std::span<std::size_t const> kept, std::size_t first_query,
                              products_tape<key_t, score_type_> const& matches) noexcept {
        std::size_t const capacity = best.size() / kept.size();
        for (std::size_t query = 0; query != kept.size(); ++query) {
            std::span<maxsim_ranked<total_t>> const ranked = best.subspan(query * capacity, kept[query]);
            std::span<key_t const> const keys = best_keys.subspan(query * capacity, capacity);
            partial_sort(ranked, ranked.size(),
                         [&](maxsim_ranked<total_t> const& first, maxsim_ranked<total_t> const& second) noexcept {
                             return better_(first.total, keys[first.candidate], second.total, keys[second.candidate]);
                         });
            std::size_t const staged = matches.staging_of(first_query + query);
            for (std::size_t rank = 0; rank != ranked.size(); ++rank)
                matches.keys[staged + rank] = keys[ranked[rank].candidate],
                                      matches.scores[staged + rank] = static_cast<score_type_>(ranked[rank].total);
        }
    }

    /** Bytes a MaxSim block of @p queries_count queries over @p rows rows of @p columns stages,
     *  each ranking up to @p capacity keys, laid out as @c maxsim_block_ carves them, and from a
     *  tile's columns of rows on, the @c tiled_fold_t past them. */
    std::size_t maxsim_staged_bytes_(std::size_t columns, std::size_t rows, std::size_t queries_count,
                                     traversal_t nominating, search_config_t config,
                                     std::size_t capacity) const noexcept {
        using slice_t = arena_allocator<byte_t>;
        std::size_t const width = nominating == traversal_t::walk_k ? config.expansion_factor : 0;
        std::size_t const bytes = staged_rows_bytes(vector_dtype_k, rows, columns) +
                                  slice_t::reserved_bytes(rows * sizeof(vector_norms_t)) +
                                  slice_t::reserved_bytes(rows * sizeof(dot_result_t)) +
                                  slice_t::reserved_bytes(queries_count * capacity * sizeof(maxsim_ranked<total_t>)) +
                                  slice_t::reserved_bytes(queries_count * capacity * sizeof(key_t)) +
                                  slice_t::reserved_bytes(queries_count * sizeof(std::size_t)) +
                                  slice_t::reserved_bytes(queries_count * sizeof(std::size_t)) +
                                  slice_t::reserved_bytes(width * sizeof(key_t)) +
                                  slice_t::reserved_bytes(width * sizeof(distance_t)) +
                                  slice_t::reserved_bytes(rows * width * sizeof(maxsim_hit_t)) +
                                  slice_t::reserved_bytes(rows * width * sizeof(maxsim_ranked<total_t>)) +
                                  slice_t::reserved_bytes(rows * sizeof(distance_t));
        if constexpr (index_t::has_tiles_k)
            if (rows >= tiles_kit_t::tile_columns_k) {
                std::size_t const block_rows = block_rows_(config.cache_bytes);
                return bytes +
                       slice_t::reserved_bytes(divide_round_up(rows, tiles_kit_t::pack_columns_k) * pack_stride_()) +
                       slice_t::reserved_bytes(block_rows * packed_bytes(vector_dtype_k, dimensions())) +
                       slice_t::reserved_bytes((block_rows + 1) * sizeof(std::size_t)) +
                       slice_t::reserved_bytes((block_rows + 1) * sizeof(key_t)) +
                       slice_t::reserved_bytes((block_rows + 1) * rows * sizeof(dot_result_t));
            }
        return bytes;
    }

    /**
     *  @brief Late interaction of the queries @p query_offsets bounds on @p traversal: every
     *      candidate folds each of its rows below @p watermarks into each query's ranking.
     *
     *  Each query stages its best @p capacity keys on @p matches, largest first, the first query as
     *  query @p first_query, and counts every key it ranked in @p scored. A walk serves one query,
     *  each of its rows ranking @p config's breadth of keys, and visits each key once in descending
     *  bound: its best walk dot per row, and the worst kept dot of every row that missed it, until
     *  a full ranking's worst total beats the next bound; under a @c traversal_t::scan_k scan every
     *  key is a candidate once for every query, at the one visible row @c leads_ picks.
     *  @p predicate hears each candidate before it folds. A tile's columns of rows or more fold
     *  through the tiles kit, fewer a quad of rows at a time. Everything lives in the traversal's
     *  staged queries, laid out as @c maxsim_staged_bytes_ counts it: the rows, their norms, one
     *  maximum per row, each query's ranking, its keys, its count and its worst slot, under walks
     *  one row's merged keys and distances, every row's hits, the keys' bounds and one gain per
     *  row, then the tiled fold. A destructive window since @p epoch answers @c contended_k before
     *  anything is written, so the caller reruns the block.
     */
    template <typename predicate_type_, typename score_type_>
    status_t maxsim_block_(rows_view_t queries, std::span<std::size_t const> query_offsets, std::size_t first_query,
                           traversal_t nominating, search_config_t config, std::size_t capacity,
                           std::span<std::size_t> scored, traversal_context_t& traversal,
                           std::span<node_id_t const> watermarks, u64_t epoch, predicate_type_& predicate,
                           products_tape<key_t, score_type_> const& matches) noexcept {
        constexpr bool filtered_k = !std::is_same_v<std::remove_cv_t<predicate_type_>, dummy_predicate_t>;
        std::size_t const queries_count = query_offsets.size() - 1, first_row = query_offsets.front();
        std::size_t const rows = query_offsets.back() - first_row;
        std::size_t const width = nominating == traversal_t::walk_k ? config.expansion_factor : 0;
        vectors_view_t const query_rows = shards_[0].index->import_rows_(queries, first_row, rows, traversal);
        bytes_span_t staged = to_writable_span(traversal.staged_queries)
                                  .subspan(staged_rows_bytes(vector_dtype_k, rows, queries.columns));
        std::span<vector_norms_t> const norms = carve_<vector_norms_t>(staged, rows);
        std::span<dot_result_t> const maxima = carve_<dot_result_t>(staged, rows);
        std::span<maxsim_ranked<total_t>> const best = carve_<maxsim_ranked<total_t>>(staged, queries_count * capacity);
        std::span<key_t> const best_keys = carve_<key_t>(staged, queries_count * capacity);
        std::span<std::size_t> const kept = carve_<std::size_t>(staged, queries_count);
        std::span<std::size_t> const worst = carve_<std::size_t>(staged, queries_count);
        std::span<key_t> const ranked = carve_<key_t>(staged, width);
        std::span<distance_t> const merged = carve_<distance_t>(staged, width);
        std::span<maxsim_hit_t> const hits = carve_<maxsim_hit_t>(staged, rows * width);
        std::span<maxsim_ranked<total_t>> const bounds = carve_<maxsim_ranked<total_t>>(staged, rows * width);
        std::span<distance_t> const row_gains = carve_<distance_t>(staged, rows);
        for (std::size_t row = 0; row != rows; ++row)
            norms[row] = shards_[0].index->norms(query_rows.row(row).as_vector());
        std::memset(kept.data(), 0, kept.size_bytes());
        std::memset(scored.data(), 0, scored.size_bytes());

        // Each query sums its own rows of a key's maxima.
        auto rank_key = [&](key_t const& key, std::span<dot_result_t const> key_maxima) noexcept {
            for (std::size_t query = 0; query != queries_count; ++query) {
                std::size_t const begin = query_offsets[query] - first_row, end = query_offsets[query + 1] - first_row;
                total_t total {};
                for (dot_result_t const maximum : key_maxima.subspan(begin, end - begin)) total += total_t(maximum);
                rank_into_(best.subspan(query * capacity, capacity), best_keys.subspan(query * capacity, capacity),
                           kept[query], worst[query], key, total);
                ++scored[query];
            }
        };
        auto score_key = [&](key_t const& key) noexcept {
            for (dot_result_t& maximum : maxima) maximum = dot_result_t::finite_min();
            std::size_t folded = 0;
            shards_.find_each(key, watermarks, [&](index_t const& shard_index, node_id_t node) noexcept {
                vector_view_t const row = shard_index.vector(node);
                fold_quads<quads_kit_t>(
                    row, shards_[0].index->norms(row), query_rows, norms, maxima,
                    [](dot_result_t& maximum, dot_result_t dot) noexcept { maximum = max_of(maximum, dot); });
                ++folded;
            });
            if (!folded) return;
            traversal.computed_distances += folded * rows;
            rank_key(key, maxima);
        };

        // Hands `visit` each candidate once: every key at the row it leads, asked a quad of rows at
        // a time and settled once per first chunk's width of rows, or a walk's keys by bytes.
        auto nominate = [&]<typename visit_type_>(visit_type_&& visit) noexcept {
            if (nominating == traversal_t::scan_k) {
                std::size_t const settled_rows = index_t::chunk_schedule_t::nodes_in(0);
                for (std::size_t shard = 0; shard != watermarks.size(); ++shard) {
                    keyed_t const& keys = shards_[shard].index->keys();
                    auto const key_of = [&keys](node_id_t node) noexcept { return keys[node]; };
                    node_predicate admits {predicate, key_of};
                    auto keep = [&](scan_payload_t node) noexcept { visit(keys[node]); };
                    for (std::size_t start = 0; start < watermarks[shard]; start += settled_rows) {
                        std::size_t const end = min_of<std::size_t>(start + settled_rows, watermarks[shard]);
                        quad<node_id_t> nodes;
                        quad<scan_payload_t> payloads;
                        std::size_t filled = 0;
                        for (std::size_t node = start; node != end; ++node) {
                            if (!leads_(shard, static_cast<node_id_t>(node), keys[node], watermarks)) continue;
                            nodes[filled] = static_cast<node_id_t>(node);
                            payloads[filled] = static_cast<scan_payload_t>(node);
                            if (++filled != quad_lanes_k) continue;
                            ask_contenders(admits, nodes, payloads, filled, keep);
                            filled = 0;
                        }
                        ask_contenders(admits, nodes, payloads, filled, keep);
                        settle_predicate(predicate, keep);
                    }
                }
                return status_t::success_k;
            }
            // Each row's walks merge into its best `width` keys, which pair with the hits by how
            // far each rose above the row's worst kept dot, the most a missed key can reach.
            std::size_t nominated = 0;
            total_t floors {};
            bool bounded = true;
            for (std::size_t row = 0; row != rows; ++row) {
                vector_view_t const query = query_rows.row(row).as_vector();
                std::size_t found = 0;
                for (std::size_t shard = 0; shard != watermarks.size(); ++shard) {
                    shard_t const& slot = shards_[shard];
                    expected<search_result_t> walked {};
                    if constexpr (!filtered_k)
                        walked = slot.index->walk_(query, dummy_fold_t {}, dummy_rescoring_t {}, traversal, config,
                                                   watermarks[shard]);
                    else {
                        keyed_t const& keys = slot.index->keys();
                        auto const key_of = [&keys](node_id_t node) noexcept { return keys[node]; };
                        node_predicate admits {predicate, key_of};
                        walked = slot.index->walk_if_(query, admits, traversal, config, watermarks[shard]);
                    }
                    if (!walked) return walked.status;
                    found = merge_search_results(to_span(traversal.results), slot.index->keys(), ranked, merged, found);
                }
                if (!found) {
                    bounded = false;
                    continue;
                }
                distance_t const farthest = merged[found - 1];
                floors += total_t(1 - farthest);
                for (std::size_t hit = 0; hit != found; ++hit)
                    hits[nominated++] = {ranked[hit], static_cast<u32_t>(row), farthest - merged[hit]};
            }

            // Grouped by key, a key's bound adds its best gain per row to each row's floor dot.
            for (distance_t& gain : row_gains) gain = 0;
            radix_group(hits.first(nominated), &maxsim_hit_t::key);
            std::size_t unique = 0;
            for (std::size_t run = 0, end = 0; run != nominated; run = end) {
                while (end != nominated && hits[end].key == hits[run].key) ++end;
                for (std::size_t hit = run; hit != end; ++hit)
                    row_gains[hits[hit].row] = max_of(row_gains[hits[hit].row], hits[hit].gain);
                total_t bound = floors;
                for (std::size_t hit = run; hit != end; ++hit)
                    bound += total_t(row_gains[hits[hit].row]), row_gains[hits[hit].row] = 0;
                hits[unique].key = hits[run].key;
                bounds[unique] = {static_cast<u32_t>(unique), bound};
                ++unique;
            }
            partial_sort(bounds.first(unique), unique,
                         [](maxsim_ranked<total_t> const& first, maxsim_ranked<total_t> const& second) noexcept {
                             return first.total > second.total;
                         });
            // The walk serves one query; once its ranking fills, bounds only fall, so the first
            // below the worst kept total ends the visits. A tiled fold ranks late, so stops later.
            for (std::size_t position = 0; position != unique; ++position) {
                if (bounded && kept[0] == capacity && best[worst[0]].total > bounds[position].total) {
                    // Skipped keys still count as scored, so `scored` sizes a call that keeps all.
                    scored[0] += unique - position;
                    break;
                }
                visit(hits[bounds[position].candidate].key);
            }
            return status_t::success_k;
        };

        if constexpr (index_t::has_tiles_k)
            if (rows >= tiles_kit_t::tile_columns_k) {
                if (failed(traversal.reserve_tiles(tiles_kit_t::stream_rows_k, tiles_kit_t::pack_columns_k, 0)))
                    return status_t::bad_alloc_k;
                tiled_fold_t fold {*this, query_rows, staged, config.cache_bytes, watermarks, traversal};
                if (status_t const nominated = nominate([&](key_t const& key) noexcept { fold.add(key, rank_key); });
                    failed(nominated))
                    return nominated;
                fold.finish(rank_key);
                if (!shards_.still_valid(epoch)) return status_t::contended_k;
                return write_ranked_(best, best_keys, kept, first_query, matches), status_t::success_k;
            }
        if (status_t const nominated = nominate(score_key); failed(nominated)) return nominated;
        // Writing is the caller's side effect, so a ranking read across a window is never written.
        if (!shards_.still_valid(epoch)) return status_t::contended_k;
        return write_ranked_(best, best_keys, kept, first_query, matches), status_t::success_k;
    }

    /** Forgets every published shard's rows and keys in place, each keeping its capacity, keeping
     *  the shard generations past the first as idle shards: what a clear, an emptied donor and a
     *  split's source become, inside their own window with every shard closed. */
    void clear_() noexcept {
        for (shard_t& shard : shards_) shard.index->forget_(static_cast<node_id_t>(shard.index->size()));
        shards_.forget_entries();
    }

    /** Closes every shard under the unique side, refusing a call in flight, a claim outstanding or
     *  a plan holding the index closed with @c contended_k; closing moves nothing a reader sees, so
     *  it moves no epoch. */
    status_t close_gated_() noexcept {
        unique_lock_t const held {gate_, std::try_to_lock};
        return held ? shards_.close() : status_t::contended_k;
    }

    /** The reachability audit behind @ref health, under the caller's executor claim and epoch. */
    expected<reachability_t> health_(executor_t& executor) noexcept {
        std::size_t const shards = shards_.size();
        growing_buffer<std::size_t, forkunion_allocator_t> cursors {shards_.allocator()};
        growing_buffer<typename index_t::health_plan_t, forkunion_allocator_t> plans {shards_.allocator()};
        if (failed(cursors.resize(shards)) || failed(plans.reserve_exactly(shards))) return {{}, status_t::bad_alloc_k};
        for (std::size_t shard = 0; shard != shards; ++shard) {
            auto [plan, status] = shards_[shard].index->plan_health();
            if (failed(status)) return {{}, status};
            plans.push_back_reserved(std::move(plan));
        }
        if (status_t const audited = sweep_reachability_(plans, to_writable_span(cursors), executor); failed(audited))
            return {{}, audited};

        reachability_t result;
        for (typename index_t::health_plan_t const& plan : plans) {
            auto const [shard_health, status] = plan.finish();
            if (failed(status)) return {{}, status};
            result += shard_health;
        }
        return {result, status_t::success_k};
    }

    /** The walk behind the single @ref search, under the caller's epoch. */
    expected<search_result_t> search_(vector_view_t query_vector, traversal_context_t& traversal,
                                      search_config_t config, std::span<key_t> keys,
                                      std::span<distance_t> distances) noexcept {
        search_result_t merged;
        std::span<node_id_t const> const watermarks = snapshot_(traversal);
        if (watermarks.empty()) return {merged, status_t::bad_alloc_k};
        status_t status = status_t::success_k;
        for (std::size_t shard = 0; shard != watermarks.size(); ++shard) {
            auto const [search_result, search_status] = shards_[shard].index->walk_(
                query_vector, dummy_fold_t {}, dummy_rescoring_t {}, traversal, config, watermarks[shard]);
            if (failed(search_status)) {
                status = search_status;
                break;
            }
            merged.visited_members += search_result.visited_members;
            merged.computed_distances += search_result.computed_distances;
            merged.restarts += search_result.restarts;
            merged.saturated_walks += search_result.saturated_walks;
            merged.count = merge_search_results(to_span(traversal.results), shards_[shard].index->keys(), keys,
                                                distances, merged.count);
        }
        return {merged, status};
    }

    /** Up to @p wanted rows under @p key exported into @p vectors, and the true total, each shard's
     *  own @c get filling the rows the earlier shards left: the body of both @ref get forms, under
     *  the caller's epoch and commit sequence. */
    std::size_t get_(key_t const& key, rows_span_t vectors, std::size_t wanted) const noexcept {
        std::size_t const room = min_of(wanted, vectors.rows);
        std::size_t total = 0;
        for (shard_t const& shard : shards_) {
            std::size_t const kept = min_of(total, room);
            total += shard.index->get(key, vectors.slice(kept, room - kept), room - kept);
        }
        return total;
    }

    /** One unreserved shard of the index's geometry placed on @p memory_domain; null when the
     *  allocation fails. */
    std::unique_ptr<index_t> build_(fu_memory_domain_id_t memory_domain) const noexcept {
        forkunion_allocator_t const allocator {memory_domain};
        return std::unique_ptr<index_t>(new (std::nothrow)
                                            index_t(kernel(), geometry_, allocator, allocator, line_bytes_));
    }

    /** One shard of a new shard generation, on the memory domain @p like sits on, bound under the
     *  shard capacity; a null index when the allocation or the reservation fails. */
    shard_t build_shard_(shard_t const& like) const noexcept {
        fu_memory_domain_id_t const memory_domain = like.index->vectors_allocator_.memory_domain;
        std::unique_ptr<index_t> index = build_(memory_domain);
        reserve_capacity_t bound {};
        bound.ceiling = shards_.shard_capacity();
        if (index && failed(index->reserve(bound).status)) index.reset();
        return shard_t {std::move(index)};
    }

    /** Sizes the caller's context of every executor thread for a walk of @p config staging
     *  @p staged_bytes of queries, seeded apart and with its counters zeroed, so the call reports
     *  only what it spent. */
    status_t size_traversals_(std::span<traversal_context_t> traversals, insert_config_t config,
                              std::size_t staged_bytes) noexcept {
        for (std::size_t thread = 0; thread != traversals.size(); ++thread) {
            traversal_context_t& traversal = traversals[thread];
            // Apart, and apart across calls: identically seeded contexts would send every
            // concurrent inserter through the same entry points.
            traversal.random_generator.state = shards_.routed() + thread;
            traversal.computed_distances = traversal.computed_distances_in_refines = 0;
            traversal.computed_distances_in_reverse_refines = traversal.visited_members = 0;
            traversal.restarts = traversal.saturated_walks = 0;
            if (status_t const sized = shards_[0].index->reserve_traversal(traversal, config); failed(sized))
                return sized;
            if (failed(traversal.reserve_queries(staged_bytes))) return status_t::bad_alloc_k;
        }
        return status_t::success_k;
    }

    using carved_traversals_t =
        growing_buffer<traversal_context_t, arena_allocator<traversal_context_t, forkunion_allocator_t>>;

    /** Rewinds every arena of @p executor once and carves one context per thread into
     *  @p traversals, its array in the first arena: what the one-shot drivers walk on, as no caller
     *  hands them contexts. The plans they drive size them. */
    status_t carve_traversals_(carved_traversals_t& traversals, executor_t& executor) noexcept {
        if (status_t const rewound = executor.reserve(0); failed(rewound)) return rewound;
        if (failed(traversals.reserve_exactly(executor.size()))) return status_t::bad_alloc_k;
        for (std::size_t thread = 0; thread != executor.size(); ++thread)
            traversals.push_back_reserved(
                traversal_context_t {arena_allocator<byte_t, forkunion_allocator_t>(executor.arena(thread))});
        return status_t::success_k;
    }

    /** Runs and finishes @p plan on contexts carved for the call: the one-shot refine, absorb and
     *  split. */
    template <typename progress_type_>
    expected<refine_report_t> drive_refine_(refine_plan_t& plan, executor_t& executor,
                                            progress_type_& on_visited) noexcept {
        carved_traversals_t traversals {arena_allocator<traversal_context_t, forkunion_allocator_t>(executor.arena(0))};
        if (status_t const carved = carve_traversals_(traversals, executor); failed(carved)) return {{}, carved};
        status_t const ran = plan.run(executor, to_writable_span(traversals), on_visited);
        auto const [result, finish_status] = plan.finish(executor, to_writable_span(traversals));
        return {result, failed(finish_status) ? finish_status : ran};
    }

#pragma region Health

    /**
     *  @brief Packs every shard of @p sources into a shard of this index under @p bound rows, as
     *      @ref shard_set::pack does with @p rank choosing among the shards with room, opening a
     *      shard generation when none has room; then sizes @p plan's part runs to match.
     *
     *  Move order numbers the source shards: this index's own first when it is among
     *  @p sources, then every other source's in order.
     *
     *  @par Statuses:
     *  - @c unsupported_k A source shard past @p bound, which would have to be cut first.
     */
    template <typename rank_type_>
    status_t plan_moves_(std::span<ram_indexes* const> sources, std::size_t bound, rank_type_&& rank,
                         refine_plan_t& plan) noexcept {
        std::size_t own = 0, count = 0;
        for (ram_indexes const* source : sources) {
            if (source == this) own = shards_.size();
            count += source->shards_.size();
        }
        if (failed(plan.moves_.resize(count))) return status_t::bad_alloc_k;
        auto source_shard = [&](std::size_t from) noexcept -> shard_t const& {
            if (from < own) return shards_[from];
            from -= own;
            std::size_t source = 0;
            for (; sources[source] == this || from >= sources[source]->shards_.size(); ++source)
                if (sources[source] != this) from -= sources[source]->shards_.size();
            return sources[source]->shards_[from];
        };
        auto domain_of = [](shard_t const& shard) noexcept { return shard.index->vectors_allocator_.memory_domain; };
        auto open = [&]() noexcept {
            // This index is sealed, so the shard generation opens closed like the rest, past the
            // seal.
            return shards_.open_shard_generation_past_seal_(
                shards_.open_shard_generation(), [this](shard_t const& built) noexcept { return build_shard_(built); });
        };
        if (status_t const packed = shards_.pack(to_writable_span(plan.moves_), own, bound, source_shard, domain_of,
                                                 rank, open);
            failed(packed))
            return packed;
        return shards_t::size_parts(to_span(plan.moves_), shards_.size(), plan.part_offsets_, plan.part_bases_);
    }

    /** Moves in every row of @p donor that @p moves, one per donor shard, name, under the closed
     *  dispensers of both indexes and the executor's claim @p refining holds: the donor's gate and
     *  every other fallible step first, then the chunk steps, which cannot fail, each copying its
     *  rows' keys, then one task per receiving shard files them, then one commit bracket publishes,
     *  and the moved shards enter the plan's parts. A donor other than this index then forgets its
     *  rows in place inside its window; this index as its own donor compacts in place, its moved
     *  shards forgetting their rows and keys inside one window around the bracket, the emptied
     *  shards staying as idle slots. */
    status_t absorb_(ram_indexes& donor, std::span<shard_move_t const> moves, executor_t& executor,
                     refine_plan_t& refining) noexcept {
        std::size_t const shards = refining.cursors_.size();
        if (!donor.size()) return status_t::success_k;
        // The donor empties once its rows move, so its gate comes before anything a refusal undoes.
        unique_lock_t const emptying {donor.gate_, std::try_to_lock};
        if (!emptying) return status_t::contended_k;

        // In place, a shard packed into itself stays and keeps its IDs.
        bool const in_place = &donor == this;
        auto moves_into = [&](std::size_t from, std::size_t into) noexcept {
            return moves[from].into == into && (!in_place || from != into);
        };

        // Every receiving shard claims all the rows it takes up front, so nothing past the claims
        // can fail and a failure releases them.
        std::size_t const donors = moves.size();
        growing_buffer<typename index_t::load_plan_t, forkunion_allocator_t> plans {shards_.allocator()};
        growing_buffer<std::size_t, forkunion_allocator_t> cursors {shards_.allocator()};
        if (failed(plans.reserve_exactly(shards)) || failed(cursors.resize(shards))) return status_t::bad_alloc_k;
        for (std::size_t into = 0; into != shards; ++into) {
            std::size_t rows = 0;
            for (std::size_t from = 0; from != donors; ++from)
                if (moves_into(from, into)) rows += donor.shards_[from].index->size();
            auto [plan, status] = shards_[into].index->plan_load(rows);
            if (failed(status)) return status;
            plans.push_back_reserved(std::move(plan));
        }

        // A step is one storage chunk of a receiving shard, which the donor parts moving there
        // fill in turn, each with its keys; a chunk may straddle several parts.
        auto copy_chunk = [&](std::size_t into, typename index_t::load_plan_t::step_t const& chunk, std::size_t,
                              nk::matrix_span<node_id_t>) noexcept -> status_t {
            iota_view<node_id_t> placed {plans[into].base, plans[into].base};
            for (std::size_t from = 0; from != donors && placed.bound < chunk.index_nodes.bound; ++from) {
                if (!moves_into(from, into)) continue;
                index_t const& part = *donor.shards_[from].index;
                iota_view<node_id_t> const source_range {0, static_cast<node_id_t>(part.size())};
                placed = {placed.bound, placed.bound + source_range.bound};
                typename index_t::load_plan_t::step_t step = chunk;
                step.index_nodes = {max_of(chunk.index_nodes.first, placed.first),
                                    min_of(chunk.index_nodes.bound, placed.bound)};
                if (step.index_nodes.first >= step.index_nodes.bound) continue;
                step.source_nodes = {step.index_nodes.first - placed.first, step.index_nodes.bound - placed.first};
                if (status_t const loaded = plans[into].load(step, part, source_range); failed(loaded)) return loaded;
            }
            return status_t::success_k;
        };
        if (status_t const copied = drain_steps_(plans, to_writable_span(cursors), executor, 0, copy_chunk);
            failed(copied))
            return copied;
        // Each receiving shard's map takes one writer, its keys invisible until its watermark
        // moves.
        auto publish_keys = [&](std::size_t into, std::size_t, std::size_t) noexcept { plans[into].publish_keys(); };
        if (failed(executor.for_tasks(shards, publish_keys)))
            for (std::size_t into = 0; into != shards; ++into) publish_keys(into, 0, 0);

        // Every step ran, so no plan refuses; the bracket shows a search and a lookup every shard's
        // rows or none. In place, the window around it makes a reader overlapping the moves retry
        // rather than miss or double a row.
        status_t published = status_t::success_k;
        if (in_place) shards_.begin_window();
        shards_.begin_commit();
        for (typename index_t::load_plan_t& plan : plans)
            if (succeeded(published)) published = plan.finish();
        if (succeeded(published)) {
            // Each moved shard becomes a part of its own in the plan, which sized every shard's
            // run.
            shards_.record_parts(donor.shards_, moves, plans, refining.part_bases_.data(),
                                 to_span(refining.part_offsets_));
            if (in_place) forget_moved_(moves);
        }
        shards_.end_commit();
        if (in_place) shards_.end_window();
        if (failed(published) || in_place) return published;

        donor.shards_.begin_window();
        donor.clear_();
        donor.shards_.end_window();
        return status_t::success_k;
    }

    /** Inside the in-place window, once every move published: forgets the rows and keys of
     *  every shard @p moves sent elsewhere, its watermark dropping to zero first. */
    void forget_moved_(std::span<shard_move_t const> moves) noexcept {
        for (std::size_t from = 0; from != moves.size(); ++from)
            if (moves[from].into != from)
                shards_[from].index->forget_(static_cast<node_id_t>(shards_[from].index->size()));
    }

    /** One reachability audit over every shard's plan: the bitsets restart, then every level
     *  sweeps every step and advances each plan once, until no level expands a node. */
    template <typename plan_type_>
    status_t sweep_reachability_(growing_buffer<plan_type_, forkunion_allocator_t>& plans,
                                 std::span<std::size_t> cursors, executor_t& executor) noexcept {
        for (plan_type_& plan : plans) plan.restart_sweep();
        auto sweep_step = [&](std::size_t shard, auto const& step, std::size_t, nk::matrix_span<node_id_t>) noexcept {
            return plans[shard].sweep(step), status_t::success_k;
        };
        for (std::size_t expanded = 1; expanded;) {
            std::memset(cursors.data(), 0, cursors.size_bytes());
            if (status_t const drained = drain_steps_(plans, cursors, executor, 0, sweep_step); failed(drained))
                return drained;
            expanded = 0;
            for (plan_type_& plan : plans) expanded += plan.advance_level();
        }
        return status_t::success_k;
    }

#pragma endregion Health

#pragma region Search

    /** Takes the shards' snapshot into @p traversal 's staged IDs, idle during any search and sized
     *  once per context as an insertion sizes them, so a read allocates nothing per call but the
     *  first after a shard generation opens; empty when they cannot hold one watermark per shard.
     *  Every walk spans the shards the snapshot counted, never the ones opened after it. */
    std::span<node_id_t const> snapshot_(traversal_context_t& traversal) const noexcept {
        while (true) {
            std::size_t const shards = shards_.size();
            if (failed(traversal.reserve_neighbors(max_of(connectivity(), shards))) ||
                nk::failed(traversal.staged_ids.resize({shards})))
                return {};
            std::span<node_id_t> const watermarks {traversal.staged_ids.data(), shards};
            if (succeeded(shards_.snapshot(watermarks).status)) return watermarks;
        }
    }

    /** A scan's ranking: each query's nearest rows merge into its keys and distances as they
     *  arrive, each one tightening the threshold once the row is full. */
    template <typename score_type_>
    struct ranked_sink {
        nk::matrix_span<key_t> keys;
        nk::matrix_span<score_type_> distances;
        std::span<std::size_t> counts;

        std::size_t staged_bytes(std::size_t) const noexcept { return 0; }

        void open(std::size_t first, std::size_t count, byte_t*) const noexcept {
            std::memset(counts.data() + first, 0, count * sizeof(std::size_t));
        }

        distance_t keep(std::size_t first, std::size_t column, shard_t const& shard, scored_node_t candidate,
                        byte_t*) const noexcept {
            std::size_t const query = first + column;
            std::span<score_type_> const distances_row = to_span(distances.row(query).as_vector());
            counts[query] = merge_search_results(std::span<scored_node_t const>(&candidate, 1), shard.index->keys(),
                                                 to_span(keys.row(query).as_vector()), distances_row, counts[query]);
            return counts[query] == distances_row.size() ? static_cast<distance_t>(distances_row.back())
                                                         : std::numeric_limits<distance_t>::max();
        }

        status_t close_shard(std::size_t, rows_view_t, shard_t const&, traversal_context_t&, byte_t*) const noexcept {
            return status_t::success_k;
        }

        status_t close(std::size_t, std::size_t, byte_t*) const noexcept { return status_t::success_k; }
    };

    /**
     *  @brief A scan's products: each query ranks one shard's rows by the finalized @c 1 - dot,
     *      then a second pass scores the at most a stride of kept rows exactly and merges their
     *      dots into the query's staged run on the tape, largest first.
     *
     *  The finalized distance loses bits a dot cast back cannot recover, so only the ranking reads
     *  it. Each query's scratch past the scan's staging holds its kept rows' fill, node IDs and
     *  distances, after one shared run of exact dots.
     */
    template <typename product_type_>
    struct products_sink {
        products_tape<key_t, product_type_> tape;
        std::span<std::size_t> counts;

        std::size_t wanted() const noexcept { return tape.stride; }
        std::size_t dots_bytes() const noexcept {
            return round_up_to_multiple<std::size_t>(wanted() * sizeof(dot_result_t), sizeof(u32_t));
        }
        std::size_t slot_bytes() const noexcept {
            return sizeof(u32_t) + wanted() * (sizeof(node_id_t) + sizeof(f32_t));
        }
        std::size_t staged_bytes(std::size_t queries) const noexcept { return dots_bytes() + queries * slot_bytes(); }

        u32_t& kept_count_(std::size_t column, byte_t* staged) const noexcept {
            return *reinterpret_cast<u32_t*>(staged + dots_bytes() + column * slot_bytes());
        }
        std::span<node_id_t> kept_ids_(std::size_t column, byte_t* staged) const noexcept {
            return {reinterpret_cast<node_id_t*>(&kept_count_(column, staged) + 1), wanted()};
        }
        std::span<f32_t> kept_distances_(std::size_t column, byte_t* staged) const noexcept {
            return {reinterpret_cast<f32_t*>(kept_ids_(column, staged).data() + wanted()), wanted()};
        }

        void open(std::size_t first, std::size_t count, byte_t* staged) const noexcept {
            std::memset(counts.data() + first, 0, count * sizeof(std::size_t));
            for (std::size_t column = 0; column != count; ++column) kept_count_(column, staged) = 0;
        }

        distance_t keep(std::size_t, std::size_t column, shard_t const&, scored_node_t candidate,
                        byte_t* staged) const noexcept {
            std::size_t const wanted = this->wanted();
            u32_t& fill = kept_count_(column, staged);
            std::span<node_id_t> const ids = kept_ids_(column, staged);
            std::span<f32_t> const distances = kept_distances_(column, staged);
            std::size_t position = fill;
            while (position && candidate.distance < distances[position - 1]) --position;
            if (position != wanted) {
                std::size_t const shifted = min_of<std::size_t>(fill, wanted - 1) - position;
                std::memmove(ids.data() + position + 1, ids.data() + position, shifted * sizeof(node_id_t));
                std::memmove(distances.data() + position + 1, distances.data() + position, shifted * sizeof(f32_t));
                ids[position] = candidate.node_id, distances[position] = candidate.distance;
                fill += fill != wanted;
            }
            return fill == wanted ? distances.back() : std::numeric_limits<distance_t>::max();
        }

        /** Scores every query's kept rows of @p shard exactly and merges them into its row, ranked
         *  by product, then empties the kept rows for the next shard. */
        status_t close_shard(std::size_t first, rows_view_t block_queries, shard_t const& shard,
                             traversal_context_t& traversal, byte_t* staged) const noexcept {
            std::span<dot_result_t> const dots {reinterpret_cast<dot_result_t*>(staged), wanted()};
            keyed_t const& shard_keys = shard.index->keys();
            for (std::size_t column = 0; column != block_queries.rows; ++column) {
                u32_t& fill = kept_count_(column, staged);
                if (!fill) continue;
                std::span<node_id_t const> const kept = kept_ids_(column, staged).first(fill);
                shard.index->dots_(shard.index->import_rows_(block_queries, column, 1, traversal).row(0).as_vector(),
                                   kept, dots.first(fill));
                std::size_t const query = first + column, staged = tape.staging_of(query);
                std::span<key_t> const keys_row = tape.keys.subspan(staged, wanted());
                std::span<product_type_> const products_row = tape.scores.subspan(staged, wanted());
                for (std::size_t rank = 0; rank != fill; ++rank)
                    counts[query] = merge_product_(shard_keys[kept[rank]], static_cast<product_type_>(dots[rank]),
                                                   keys_row, products_row, counts[query]);
                fill = 0;
            }
            return status_t::success_k;
        }

        status_t close(std::size_t, std::size_t, byte_t*) const noexcept { return status_t::success_k; }

        /** Inserts @p key at @p product into the @p kept leading pairs of a row ranked largest
         *  first, dropping the smallest of a full row; answers the pairs kept after. */
        static std::size_t merge_product_(key_t key, product_type_ product, std::span<key_t> keys_row,
                                          std::span<product_type_> products_row, std::size_t kept) noexcept {
            std::size_t const width = keys_row.size();
            std::size_t position = kept;
            while (position && products_row[position - 1] < product) --position;
            if (position == width) return kept;
            std::size_t const shifted = min_of<std::size_t>(kept, width - 1) - position;
            std::memmove(keys_row.data() + position + 1, keys_row.data() + position, shifted * sizeof(key_t));
            std::memmove(products_row.data() + position + 1, products_row.data() + position,
                         shifted * sizeof(product_type_));
            keys_row[position] = key, products_row[position] = product;
            return kept + (kept != width);
        }
    };

    /**
     *  @brief A scan's attention: every kept row folds its value row into each query row's online
     *      softmax as it arrives, every query row a head of its own, and the rows normalize once
     *      the block's last shard is scanned.
     *
     *  A logit reads the finalized @c 1 - dot back, within the distance's own rounding. Each
     *  query row's running state and @c f32 sums sit past the scan's staging, after the block's
     *  first refusal, until the block exports them.
     */
    template <typename value_lookup_type_>
    struct attention_sink {
        std::remove_reference_t<value_lookup_type_>& value_lookup;
        f32_t score_scale;
        rows_span_t outputs;
        std::span<softmax_state_t> carry;

        std::span<std::size_t> counts {};

        static constexpr std::size_t status_bytes_k = sizeof(u64_t);
        std::size_t slot_bytes() const noexcept { return sizeof(softmax_state_t) + outputs.columns * sizeof(f32_t); }
        std::size_t staged_bytes(std::size_t queries) const noexcept { return status_bytes_k + queries * slot_bytes(); }

        status_t& kept_status_(byte_t* staged) const noexcept { return *reinterpret_cast<status_t*>(staged); }
        softmax_state_t& softmax_state_(std::size_t column, byte_t* staged) const noexcept {
            return *reinterpret_cast<softmax_state_t*>(staged + status_bytes_k + column * slot_bytes());
        }
        f32_t* value_sums_(std::size_t column, byte_t* staged) const noexcept {
            return reinterpret_cast<f32_t*>(&softmax_state_(column, staged) + 1);
        }

        void open(std::size_t, std::size_t count, byte_t* staged) const noexcept {
            kept_status_(staged) = status_t::success_k;
            for (std::size_t column = 0; column != count; ++column) {
                softmax_state_(column, staged) = {std::numeric_limits<f32_t>::lowest(), 0};
                std::memset(value_sums_(column, staged), 0, outputs.columns * sizeof(f32_t));
            }
        }

        distance_t keep(std::size_t, std::size_t column, shard_t const& shard, scored_node_t candidate,
                        byte_t* staged) const noexcept {
            status_t& status = kept_status_(staged);
            if (failed(status)) return std::numeric_limits<distance_t>::max();
            keyed_t const& shard_keys = shard.index->keys();
            key_t const key = shard_keys[candidate.node_id];
            vector_view_t row;
            value_lookup(std::span<key_t const, 1>(&key, 1), std::span<vector_view_t, 1>(&row, 1));
            if (row.size() != outputs.columns) {
                status = status_t::shape_mismatch_k;
                return std::numeric_limits<distance_t>::max();
            }
            softmax_state_t& state = softmax_state_(column, staged);
            f32_t const logit = score_scale * std::numbers::log2e_v<f32_t> *
                                (1 - static_cast<f32_t>(candidate.distance));
            f32_t const maximum = max_of(state.maximum_score, logit);
            // The new row's weight and the running sums' correction, exponentiated in one tile.
            std::array<f32_t, 8> weights {logit - maximum}, corrections {state.maximum_score - maximum};
            elementwise_kit_t::exp2(weights);
            elementwise_kit_t::exp2(corrections);
            state.partition = state.partition * corrections[0] + weights[0];
            state.maximum_score = maximum;
            char const* const row_bytes = row.byte_data();
            values_step<scalar_t, elementwise_kit_t>(value_sums_(column, staged), 1, outputs.columns, &row_bytes,
                                                     &weights, corrections, 1);
            return std::numeric_limits<distance_t>::max();
        }

        status_t close_shard(std::size_t, rows_view_t, shard_t const&, traversal_context_t&, byte_t*) const noexcept {
            return status_t::success_k;
        }

        /** Divides every query row's sums by its partition, exports the row and its natural-log
         *  state, unless a lookup refused. */
        status_t close(std::size_t first, std::size_t count, byte_t* staged) const noexcept {
            if (failed(kept_status_(staged))) return kept_status_(staged);
            for (std::size_t column = 0; column != count; ++column) {
                softmax_state_t& state = softmax_state_(column, staged);
                f32_t* const sums = value_sums_(column, staged);
                f32_t const normalizer = state.partition > 0 ? 1 / state.partition : 0, zero = 0;
                [[maybe_unused]] nk_status_t const scaled = elementwise_kit_t::scale(sums, outputs.columns, &normalizer,
                                                                                     &zero, sums, nullptr);
                assert(scaled == nk_success_k && "The serial tier scales on every CPU");
                state.maximum_score *= std::numbers::ln2_v<f32_t>;
                export_row<elementwise_kit_t>(sums, nk_f32_k, outputs, first + column);
                if (!carry.empty()) carry[first + column] = state;
            }
            return status_t::success_k;
        }
    };

    /** The scan behind @c search_if: refuses a mismatched ranking, then writes every query's
     *  nearest rows through a @ref ranked_sink. */
    template <typename score_type_, typename predicate_type_>
    expected<search_result_t> scan_ranked_(rows_view_t queries, executor_t& executor,
                                           std::span<traversal_context_t> traversals, search_config_t config,
                                           nk::matrix_span<key_t> keys, nk::matrix_span<score_type_> distances,
                                           std::span<std::size_t> counts, predicate_type_&& predicate) noexcept {
        std::size_t const queries_count = queries.rows;
        if (shards_.empty()) return {{}, status_t::uninitialized_k};
        if (counts.size() != queries_count) return {{}, status_t::config_mismatch_k};
        if (status_t const refusal = rankings_refusal(queries_count, keys, distances); failed(refusal))
            return {{}, refusal};
        std::memset(counts.data(), 0, counts.size_bytes());
        if (!queries_count || !keys.extent(1)) return {{}, status_t::success_k};
        return scan_rows_(queries, executor, traversals, config, ranked_sink<score_type_> {keys, distances, counts},
                          predicate);
    }

    /**
     *  @brief The exact scan behind a ranking, products and attention: every entry published when
     *      a query's block starts is scored, and every row beating a query's threshold goes
     *      through @p sink.
     *
     *  Each block of queries takes a thread and its traversal, scans every shard below one snapshot
     *  of the watermarks, hands each arriving row to @p sink with the shard, and closes the shard,
     *  then the block. A block at least one tile wide, on a platform with a matrix engine for the
     *  scalar, streams through it a pack at a time, in blocks of rows sized by @p config's cache
     *  bytes; a single query scores four rows at a time. Either way the shards' own kit finalizes
     *  the distances, so they match the walk's. The sink's scratch sits in each traversal's staged
     *  queries past what the scan stages, so nothing allocates once warm. The callers placed a
     *  shard, checked a non-empty batch and zeroed the sink's counts.
     */
    template <typename sink_type_, typename predicate_type_>
    expected<search_result_t> scan_rows_(rows_view_t queries, executor_t& executor,
                                         std::span<traversal_context_t> traversals, search_config_t config,
                                         sink_type_ const& sink, predicate_type_&& predicate) noexcept {
        std::size_t const queries_count = queries.rows;
        std::size_t const width = index_t::scan_width(queries_count, executor.size());
        std::size_t const scan_bytes = round_up_to_multiple<std::size_t>(shards_[0].index->scan_staged_bytes(width),
                                                                         default_alignment_k);
        auto scan_block = [&](std::size_t block, std::size_t thread, traversal_context_t& traversal,
                              std::span<node_id_t const> watermarks, u64_t) noexcept {
            std::size_t const first = block * width;
            rows_view_t const block_queries = queries.slice(first, min_of(width, queries_count - first));
            byte_t* const staged = traversal.staged_queries.data() + scan_bytes;
            // A rerun block starts its sink over.
            sink.open(first, block_queries.rows, staged);
            for (std::size_t shard = 0; shard != watermarks.size(); ++shard) {
                shard_t const& slot = shards_[shard];
                auto keep = [&](std::size_t column, scored_node_t const candidate) noexcept {
                    return sink.keep(first, column, slot, candidate, staged);
                };
                status_t scanned = status_t::success_k;
                if constexpr (std::is_same_v<std::decay_t<predicate_type_>, dummy_predicate_t>)
                    scanned = slot.index->scan_(block_queries, dummy_predicate_t {}, keep, traversal, config,
                                                watermarks[shard]);
                else {
                    keyed_t const& shard_keys = slot.index->keys();
                    auto const key_of = [&shard_keys](node_id_t node) noexcept { return shard_keys[node]; };
                    scanned = slot.index->scan_(block_queries,
                                                node_predicate {thread_predicate(predicate, thread), key_of}, keep,
                                                traversal, config, watermarks[shard]);
                }
                if (failed(scanned)) return scanned;
                if (status_t const closed = sink.close_shard(first, block_queries, slot, traversal, staged);
                    failed(closed))
                    return closed;
            }
            return sink.close(first, block_queries.rows, staged);
        };
        return for_each_row_(divide_round_up(queries_count, width), executor, traversals, search_config_t {},
                             scan_bytes + sink.staged_bytes(width), scan_block, sink.counts, predicate,
                             scan_round_capacity<typename index_t::tiles_kit_t>());
    }

    /** The batched @ref search_if under @p heads, which only a @c dummy_predicate_t accompanies. */
    template <typename score_type_, typename heads_type_, typename predicate_type_>
    expected<search_result_t> search_rows_(rows_view_t queries, heads_type_ const& heads, executor_t& executor,
                                           std::span<traversal_context_t> traversals, search_config_t config,
                                           nk::matrix_span<key_t> keys, nk::matrix_span<score_type_> distances,
                                           std::span<std::size_t> counts, predicate_type_&& predicate) noexcept {
        std::size_t const queries_count = queries.rows;
        if (shards_.empty()) return {{}, status_t::uninitialized_k};
        if (counts.size() != queries_count) return {{}, status_t::config_mismatch_k};
        if (status_t const refusal = rankings_refusal(queries_count, keys, distances); failed(refusal))
            return {{}, refusal};
        if (!queries_count) return {{}, status_t::success_k};

        auto search_query = [&](std::size_t query, std::size_t thread, traversal_context_t& traversal,
                                std::span<node_id_t const> watermarks, u64_t) noexcept {
            std::span<key_t> const keys_row = to_span(keys.row(query).as_vector());
            std::span<score_type_> const distances_row = to_span(distances.row(query).as_vector());
            vector_view_t const query_vector =
                shards_[0].index->import_rows_(queries, query, 1, traversal).row(0).as_vector();
            std::size_t found_count = 0;
            status_t refusal = status_t::success_k;
            for (std::size_t shard = 0; shard != watermarks.size(); ++shard) {
                shard_t const& slot = shards_[shard];
                expected<search_result_t> found {};
                if constexpr (!std::is_same_v<heads_type_, dummy_heads_t>)
                    found = slot.index->walk_(query_vector, dummy_fold_t {},
                                              indexer_rescoring<quads_kit_t> {heads.row(query)}, traversal, config,
                                              watermarks[shard]);
                else if constexpr (std::is_same_v<std::decay_t<predicate_type_>, dummy_predicate_t>)
                    found = slot.index->walk_(query_vector, dummy_fold_t {}, dummy_rescoring_t {}, traversal, config,
                                              watermarks[shard]);
                else {
                    keyed_t const& shard_keys = slot.index->keys();
                    auto const key_of = [&shard_keys](node_id_t node) noexcept { return shard_keys[node]; };
                    node_predicate admits {thread_predicate(predicate, thread), key_of};
                    found = slot.index->walk_if_(query_vector, admits, traversal, config, watermarks[shard]);
                }
                if (!found) {
                    refusal = found.status;
                    break;
                }
                found_count = merge_search_results(to_span(traversal.results), slot.index->keys(), keys_row,
                                                   distances_row, //
                                                   found_count);
            }
            counts[query] = found_count;
            return refusal;
        };
        return for_each_row_(queries_count, executor, traversals, config,
                             staged_rows_bytes(vector_dtype_k, 1, queries.columns), search_query, counts, predicate,
                             index_t::round_capacity(shards_[0].index->connectivity()));
    }

    /** Claims @p executor, sizes the caller's @p traversals, one per thread, with @p staged_bytes
     *  of staged queries, runs @p per_row over every row with its thread, that thread's traversal,
     *  a fresh snapshot of the watermarks and the epoch read before it, and sums the threads'
     *  telemetry & the rows' @p counts; a row that refuses stops itself, and the first refusal is
     *  the one kept. A row a destructive window overlapped reruns once on a fresh snapshot, then
     *  refuses with @c contended_k, so @p per_row writes its row's outputs whole each time. An
     *  aggregating @p predicate reserves each thread's buffers for @p candidates. Nothing here
     *  rewinds an arena, so whatever the caller carved there lives through the call. */
    template <typename per_row_type_, typename predicate_type_ = dummy_predicate_t>
    expected<search_result_t> for_each_row_(std::size_t rows_count, executor_t& executor,
                                            std::span<traversal_context_t> traversals, search_config_t config,
                                            std::size_t staged_bytes, per_row_type_&& per_row,
                                            std::span<std::size_t const> counts, predicate_type_&& predicate = {},
                                            std::size_t candidates = 0) noexcept {
        // Only `scan_rows_` scans, and it drives the rows under a walk's sizing.
        if (config.traversal != traversal_t::walk_k) return {{}, status_t::unsupported_k};
        if (traversals.size() != executor.size()) return {{}, status_t::config_mismatch_k};
        auto [executor_claim, claim_status] = executor.claim();
        if (failed(claim_status)) return {{}, claim_status};
        if (status_t const sized = size_traversals_(traversals, insert_config_t {config.expansion_factor},
                                                    staged_bytes);
            failed(sized))
            return {{}, sized};
        if (status_t const opened = open_predicate(predicate, executor); failed(opened)) return {{}, opened};
        for (std::size_t thread = 0; thread != traversals.size(); ++thread)
            if (failed(reserve_predicate(thread_predicate(predicate, thread), candidates)))
                return {{}, status_t::bad_alloc_k};

        status_t first_refusal = status_t::success_k;
        alignas(default_alignment_k) std::size_t live_rows = 0;
        std::size_t const filling = filling_nodes<std::size_t>(shards_[0].index->non_zero_expansion_factor(config),
                                                               shards_[0].index->connectivity());
        auto each_row = [&](std::size_t row, std::size_t thread, std::size_t) noexcept {
            traversal_context_t& traversal = traversals[thread];
            std::size_t const live = atomic_ref<std::size_t>(live_rows).fetch_add(1, std::memory_order_relaxed) + 1;
            traversal.walk_width = walk_width(executor.size(), live, rows_count - row, filling);
            status_t refusal = status_t::contended_k;
            for (std::size_t attempt = 0; attempt != 2; ++attempt) {
                u64_t const epoch = shards_.read_epoch();
                std::span<node_id_t const> const watermarks = snapshot_(traversal);
                status_t const ran = !watermarks.empty() ? per_row(row, thread, traversal, watermarks, epoch)
                                                         : status_t::bad_alloc_k;
                if (!shards_.still_valid(epoch)) continue;
                refusal = ran;
                break;
            }
            atomic_ref<std::size_t>(live_rows).sub(1, std::memory_order_relaxed);
            if (succeeded(refusal)) return;
            status_t expected_status = status_t::success_k;
            atomic_ref<status_t>(first_refusal)
                .compare_exchange_strong(expected_status, refusal, std::memory_order_relaxed,
                                         std::memory_order_relaxed);
        };
        if (status_t const fanned = executor.for_tasks(rows_count, each_row); failed(fanned)) return {{}, fanned};

        search_result_t merged;
        for (std::size_t const count : counts) merged.count += count;
        for (traversal_context_t const& traversal : traversals) {
            merged.visited_members += traversal.visited_members;
            merged.computed_distances += traversal.computed_distances;
            merged.restarts += traversal.restarts;
            merged.saturated_walks += traversal.saturated_walks;
        }
        return {merged, first_refusal};
    }

#pragma endregion Search

#pragma region Checkpoints

    /** One slab of neighbor IDs from @p arena - the scratch a save chunk rebases through; empty
     *  when the arena is short. */
    nk::matrix_span<node_id_t> scratch_of_(typename executor_t::arena_t& arena,
                                           std::size_t scratch_rows) const noexcept {
        std::size_t const connectivity = this->connectivity();
        std::byte* bytes = arena.allocate(scratch_rows * connectivity * sizeof(node_id_t), alignof(node_id_t));
        if (!bytes) return {};
        return {reinterpret_cast<node_id_t*>(bytes), scratch_rows, connectivity};
    }

    /** The home shard of @p thread : the first shard on its memory domain, so its drains start
     *  there, or a spread over the shards when none sits there. */
    std::size_t pick_home_shard_(std::size_t thread, executor_t const& executor) const noexcept {
        std::size_t const shards = shards_.size();
        for (std::size_t shard = 0; shard != shards; ++shard)
            if (shards_[shard].index->vectors_allocator_.memory_domain == executor.memory_domain(thread)) return shard;
        return thread % shards;
    }

    /**
     *  @brief Fans every plan's chunks out over the executor's threads: each thread drains its
     *      home shard first, then helps the rest, one atomic cursor per shard being the whole
     *      schedule.
     *
     *  @p per_step runs one step on the thread that pulled it, with that thread's scratch -
     *  @p scratch_rows of neighbor IDs, none when zero - and answers a status; the first failure
     *  wins.
     *
     *  @p cursors is the caller's, one entry per plan, so an audit reuses one across its levels
     *  rather than allocating per level.
     */
    template <typename plan_type_, typename per_step_type_>
    status_t drain_steps_(growing_buffer<plan_type_, forkunion_allocator_t>& plans, std::span<std::size_t> cursors,
                          executor_t& executor, std::size_t scratch_rows, per_step_type_& per_step) noexcept {
        std::size_t const shards = plans.size();
        if (cursors.size() != shards) return status_t::config_mismatch_k;
        status_t outcome = status_t::success_k;
        auto drain = [&](std::size_t thread, std::size_t) noexcept {
            nk::matrix_span<node_id_t> scratch;
            if (scratch_rows) {
                scratch = scratch_of_(executor.arena(thread), scratch_rows);
                if (scratch.empty()) {
                    status_t expected_status = status_t::success_k;
                    atomic_ref<status_t>(outcome).compare_exchange_strong(
                        expected_status, status_t::bad_alloc_k, std::memory_order_relaxed, std::memory_order_relaxed);
                    return;
                }
            }
            std::size_t const home = pick_home_shard_(thread, executor);
            for (std::size_t visited = 0; visited != shards; ++visited) {
                std::size_t const shard = (home + visited) % shards;
                atomic_ref<std::size_t> const cursor {cursors[shard]};
                auto const steps = plans[shard].steps();
                // The failure check comes before the claim, so every claimed step runs and the
                // cursor says where a stopped plan stopped.
                while (!failed(atomic_ref<status_t>(outcome).load(std::memory_order_relaxed))) {
                    std::size_t const position = cursor.fetch_add(1, std::memory_order_relaxed);
                    if (position >= steps.size()) break;
                    status_t const stepped = per_step(shard, steps[position], thread, scratch);
                    if (succeeded(stepped)) continue;
                    status_t expected_status = status_t::success_k;
                    atomic_ref<status_t>(outcome).compare_exchange_strong(
                        expected_status, stepped, std::memory_order_relaxed, std::memory_order_relaxed);
                    return;
                }
            }
        };
        if (status_t const fanned = executor.for_threads(drain); failed(fanned)) return fanned;
        return atomic_ref<status_t>(outcome).load(std::memory_order_relaxed);
    }

    /** Fans @p plan 's steps out over the executor's threads: each thread drains the run of the
     *  shard it is home to first, then helps the rest, one atomic cursor per shard being the whole
     *  schedule. A save writes each step with the thread's scratch of neighbor IDs; a load reads
     *  straight out of its part. The first failure wins and stops the rest. */
    template <typename plan_type_>
    status_t drain_steps_(plan_type_& plan, executor_t& executor) noexcept {
        std::size_t const shards = shards_.size();
        auto const steps = plan.steps();
        constexpr bool saving_k = requires(plan_type_& planned, typename plan_type_::step_t const& step,
                                           nk::matrix_span<node_id_t> scratch) { planned.save(step, scratch); };
        std::size_t const scratch_rows = saving_k
                                             ? power_of_two<std::size_t>(index_t::chunk_schedule_t::slab_nodes_log2_k)
                                             : 0;

        auto [executor_claim, claim_status] = executor.claim();
        if (failed(claim_status)) return claim_status;
        if (scratch_rows)
            if (status_t const sized = executor.reserve(scratch_rows * connectivity() * sizeof(node_id_t));
                failed(sized))
                return sized;
        growing_buffer<std::size_t, forkunion_allocator_t> cursors {shards_.allocator()};
        if (failed(cursors.resize(shards))) return status_t::bad_alloc_k;
        status_t outcome = status_t::success_k;
        auto drain = [&](std::size_t thread, std::size_t) noexcept {
            nk::matrix_span<node_id_t> scratch;
            if (scratch_rows) {
                scratch = scratch_of_(executor.arena(thread), scratch_rows);
                if (scratch.empty()) {
                    status_t expected_status = status_t::success_k;
                    atomic_ref<status_t>(outcome).compare_exchange_strong(
                        expected_status, status_t::bad_alloc_k, std::memory_order_relaxed, std::memory_order_relaxed);
                    return;
                }
            }
            std::size_t const home = pick_home_shard_(thread, executor);
            for (std::size_t visited = 0; visited != shards; ++visited) {
                std::size_t const shard = (home + visited) % shards;
                iota_view<std::size_t> const own = plan.steps_of(shard);
                atomic_ref<std::size_t> const cursor {cursors[shard]};
                for (std::size_t taken = cursor.fetch_add(1, std::memory_order_relaxed); taken < own.size();
                     taken = cursor.fetch_add(1, std::memory_order_relaxed)) {
                    if (failed(atomic_ref<status_t>(outcome).load(std::memory_order_relaxed))) return;
                    status_t stepped = status_t::success_k;
                    if constexpr (saving_k) stepped = plan.save(steps[own.first + taken], scratch);
                    else stepped = plan.load(steps[own.first + taken]);
                    if (succeeded(stepped)) continue;
                    status_t expected_status = status_t::success_k;
                    atomic_ref<status_t>(outcome).compare_exchange_strong(
                        expected_status, stepped, std::memory_order_relaxed, std::memory_order_relaxed);
                    return;
                }
            }
        };
        if (status_t const fanned = executor.for_threads(drain); failed(fanned)) return fanned;
        return atomic_ref<status_t>(outcome).load(std::memory_order_relaxed);
    }

#pragma endregion Checkpoints

    /** The gate every mutator holds shared for one call and every destructive window takes alone,
     *  through @c try_lock, so a window never waits; readers never touch it. */
    using shared_mutex_t = shared_mutex<wait_type_, atomic_ref_template_>;
    using shared_lock_t = std::shared_lock<shared_mutex_t>;
    using unique_lock_t = std::unique_lock<shared_mutex_t>;

    /** The shards, their routing, the commit word every reader snapshots by and the epoch every
     *  reader brackets itself with. */
    shards_t shards_;

    /** The distance kind every shard this index builds or accepts walks by. */
    nk_kernel_kind_t kernel_;

    /** The geometry every shard this index builds or accepts keeps. */
    geometry_t geometry_;

    /** Held while a planned batch dispenses its shards' ranges, so no two batches' ranges cross
     *  order between the shards and each can wait for its turns outside the commit bracket. */
    fu::spin_mutex_t dispensing_;

    /** Shared by every mutator for one call, alone by every destructive window; never swapped. */
    alignas(default_alignment_k) shared_mutex_t gate_;

    /** The line every shard this index builds prefetches by. */
    std::size_t line_bytes_;
};

/**
 *  @brief A reader's walk over the entries of one @c ram_indexes as its shards stood at open:
 *      every entry as stored, shard after shard, or breadth-first from the start entries, each
 *      entry once, each with the keys it links to.
 *
 *  Every call names the index it walks, which must be the one the cursor opened on. One shard
 *  cursor per shard stages each entry's neighbors, so the walk never blocks a writer, and links
 *  stay inside a shard, so each resolves to its key through the shard it leaves. A breadth-first
 *  walk reaches the nodes below each shard cursor's snapshot at open; a stored walk refreshes each
 *  shard's watermark as it enters that shard. The walk holds only under the epoch the cursor
 *  opened in: once a destructive window moved it, every pull answers @c contended_k.
 */
template <typename indexes_type_>
struct ram_indexes_cursor {
    using indexes_t = indexes_type_;
    using index_t = typename indexes_t::index_t;
    using key_t = typename indexes_t::key_t;
    using node_id_t = typename index_t::node_id_t;
    using walk_t = breadth_first_walk<node_id_t>;
    using shard_node_t = typename walk_t::shard_node_t;
    using shard_cursor_t = ram_cursor<index_t, synchronization_t::concurrent_k, forkunion_allocator_t>;

    /** Where a pull gathers its entries' links, in the caller's arena: the linked keys end to end,
     *  and the offset where each entry's run starts. */
    using links_allocator_t = arena_allocator<byte_t, forkunion_allocator_t>;
    using links_keys_t = growing_buffer<key_t, links_allocator_t>;
    using links_offsets_t = growing_buffer<std::size_t, links_allocator_t>;

    explicit ram_indexes_cursor(forkunion_allocator_t allocator) noexcept
        : shard_cursors_(allocator), walk_(allocator), allocator_(allocator) {}

    /**
     *  @brief Opens the walk over every shard of @p indexes under the epoch it reads now; any
     *      @p starts make it breadth-first, each key starting from its lowest node in the lowest
     *      shard holding it, and a key held nowhere starting nothing.
     *
     *  @par Statuses:
     *  - @c bad_alloc_k The shard cursors, the visited bits or the frontier found no memory.
     */
    status_t open(indexes_t const& indexes, std::span<key_t const> starts) noexcept {
        epoch_ = indexes.read_epoch();
        std::size_t const count = indexes.shards_count();
        if (failed(shard_cursors_.reserve_exactly(count))) return status_t::bad_alloc_k;
        for (std::size_t shard = 0; shard != count; ++shard) {
            shard_cursors_.push_back_reserved(shard_cursor_t {indexes.shard(shard)});
            if (status_t const reserved = shard_cursors_.back().reserve(allocator_); failed(reserved)) return reserved;
        }
        if (starts.empty()) return status_t::success_k;
        order_ = cursor_order_t::breadth_first_k;
        growing_buffer<std::size_t, forkunion_allocator_t> visible {allocator_};
        if (failed(visible.resize_uninitialized(count))) return status_t::bad_alloc_k;
        for (std::size_t shard = 0; shard != count; ++shard) visible[shard] = shard_cursors_[shard].visible_count();
        if (status_t const opened = walk_.open(to_span(visible)); failed(opened)) return opened;
        for (key_t const& key : starts)
            for (std::size_t shard = 0; shard != count; ++shard) {
                node_id_t const visible_count = shard_cursors_[shard].visible_count();
                node_id_t lowest = visible_count;
                indexes.shard(shard).find_each(key, [&](node_id_t node) noexcept { lowest = min_of(lowest, node); });
                if (lowest == visible_count) continue;
                if (status_t const visited = walk_.visit(static_cast<u32_t>(shard), lowest); failed(visited))
                    return visited;
                break;
            }
        return status_t::success_k;
    }

    /**
     *  @brief Writes up to @p keys.size() next entries' keys into @p keys, gathering each one's
     *      links into @p links_keys and @p links_offsets when both are named, and answers the
     *      number of keys written.
     *
     *  @par Statuses:
     *  - @c bad_alloc_k A link or a newly visited entry found no memory; the entries written stand.
     *  - @c contended_k A destructive window of @p indexes opened since the walk did; none stand.
     */
    expected<std::size_t> pull(indexes_t const& indexes, std::span<key_t> keys, links_keys_t* links_keys,
                               links_offsets_t* links_offsets) noexcept {
        std::size_t written = 0;
        shard_node_t entry;
        while (written != keys.size() && next_(entry)) {
            shard_cursor_t& cursor = shard_cursors_[entry.shard];
            index_t const& shard = indexes.shard(entry.shard);
            if (status_t const sought = cursor.seek(entry.node); failed(sought)) return {written, sought};
            std::span<node_id_t const> const neighbors = cursor.neighbors();
            if (links_keys && (failed(links_keys->reserve(links_keys->size() + neighbors.size())) ||
                               failed(links_offsets->reserve(links_offsets->size() + 1))))
                return {written, status_t::bad_alloc_k};
            for (node_id_t const neighbor : neighbors) {
                // Breadth-first never refreshes, so the bits span the shard cursors' snapshots.
                if (order_ == cursor_order_t::breadth_first_k && neighbor < cursor.visible_count())
                    if (status_t const visited = walk_.visit(entry.shard, neighbor); failed(visited))
                        return {written, visited};
                if (!links_keys) continue;
                expected<key_t> const linked = shard.key_of(neighbor);
                if (succeeded(linked.status)) links_keys->push_back_reserved(linked.value);
            }
            if (links_keys) links_offsets->push_back_reserved(links_keys->size());
            keys[written++] = shard.key_of(entry.node).value;
        }
        // Retired shards keep their records, so a pull that ran past a window read nothing freed.
        if (!indexes.still_valid(epoch_)) return {0, status_t::contended_k};
        return {written, status_t::success_k};
    }

  private:

    /** Takes the next entry into @p entry, answering false once the walk ran out. */
    bool next_(shard_node_t& entry) noexcept {
        if (order_ == cursor_order_t::breadth_first_k) return walk_.expand(entry);
        while (next_stored_.shard != shard_cursors_.size()) {
            if (next_stored_.node < shard_cursors_[next_stored_.shard].visible_count()) {
                entry = next_stored_;
                ++next_stored_.node;
                return true;
            }
            next_stored_ = {next_stored_.shard + 1, 0};
            if (next_stored_.shard != shard_cursors_.size()) shard_cursors_[next_stored_.shard].refresh();
        }
        return false;
    }

    /** One per shard, in placement order, each over the snapshot it took at open. */
    growing_buffer<shard_cursor_t, forkunion_allocator_t> shard_cursors_;

    /** The nodes visited and the frontier of a breadth-first walk; untouched by a stored one. */
    walk_t walk_;

    forkunion_allocator_t allocator_;
    cursor_order_t order_ = cursor_order_t::stored_k;

    /** As stored: the entry the next pull starts at. */
    shard_node_t next_stored_ {};

    /** The epoch the walk opened under; a pull after a window since answers contended. */
    u64_t epoch_ = 0;
};

} // namespace unum::usearch::dense
