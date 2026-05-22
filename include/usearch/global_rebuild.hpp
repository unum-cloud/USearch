/**
 *  @file       global_rebuild.hpp
 *  @author     Mikhail Chichvarin
 *  @brief      Non-blocking @b global-rebuild orchestrator for `index_dense_gt`.
 *  @date       May 22, 2026
 *
 *  @section Overview
 *
 *  The point of this adapter is @b durable persistence: writing the HNSW
 *  structure to disk so it survives a process restart or machine reboot,
 *  @b without a stop-the-world save. A plain `save` blocks the index for the
 *  whole flush - for a large graph that is a long window during which no
 *  reads or writes are served. `global_rebuild_gt` removes that window.
 *
 *  Persisting a live, mutating graph node-by-node is inherently racy, so the
 *  adapter persists a structurally @b frozen copy instead. It first rebuilds
 *  the index into a fresh "shadow" peer (reconstructing the graph from
 *  scratch, which also compacts away deleted slots and stale edges), then
 *  streams that frozen shadow to disk. Both phases run in small, budgeted
 *  steps, so the live "primary" index keeps serving reads and writes
 *  throughout:
 *
 *      1. `phase_migrating` - a fresh, empty "shadow" peer is built by
 *         re-inserting the primary's key-set as it stood at `begin()`. This is
 *         the actual graph reconstruction.
 *      2. `phase_saving` - the now-complete, structurally @b frozen shadow is
 *         streamed to disk through the resumable `save_to_stream`, a bounded
 *         chunk per step.
 *      3. `phase_done` - the temp file is atomically renamed onto the
 *         destination, the shadow released, tombstoned removals replayed.
 *
 *  Routing of concurrent mutations, matching the design agreed for this work:
 *
 *      * `add`    - always applied to the primary. New keys land in higher
 *                   slots; they are simply not part of the point-in-time
 *                   snapshot being rebuilt. The primary is never frozen.
 *      * `remove` - allowed only when it cannot break an in-flight save. While
 *                   a rebuild is active the physical removal is @b tombstoned
 *                   (deferred) and replayed on the primary once `phase_done`
 *                   is reached, keeping the on-disk snapshot exactly equal to
 *                   the `begin()` generation.
 *
 *  Because the file is only ever streamed from the shadow - which stops
 *  receiving writes before `phase_saving` begins - the resumable
 *  `save_to_stream` always sees a structurally frozen target, as it requires.
 *
 *  @section Crash safety
 *
 *  The rebuild streams into a @b temporary file (`<path>.tmp`) and only
 *  `rename`s it onto the destination once the whole file is complete. Until
 *  that final rename - atomic on POSIX - the destination still holds the
 *  previous index untouched. So a process kill at @b any point during a
 *  rebuild never corrupts the on-disk index: you are left with either the
 *  previous complete file or the new complete file, never a truncated one.
 *  An abandoned rebuild's temp file is discarded by the destructor.
 *
 *  Note this is crash safety for the @b destination file, not resumability
 *  across a restart: the continuation cursor lives in RAM, so a killed
 *  rebuild must be restarted from `begin`, not continued.
 *
 *  @section Memory
 *
 *  The shadow is a second HNSW @b graph, but @b not a second copy of the
 *  vectors: it is built with `add(..., copy_vector = false)`, so every shadow
 *  node references the primary's stored vector bytes (`index_dense_gt::
 *  vector_data`) instead of duplicating them. The extra RAM held during a
 *  rebuild is therefore one graph, not a full `vectors + graph` clone - for
 *  typical embedding dimensions the peak overhead is a fraction of the index,
 *  not a doubling. This is safe because the primary outlives the shadow and
 *  its existing vector bytes stay put for the whole rebuild (concurrent `add`s
 *  only append, `remove`s are deferred). The shadow is released at
 *  `phase_done`, returning even that overhead.
 */
#ifndef UNUM_USEARCH_GLOBAL_REBUILD_HPP
#define UNUM_USEARCH_GLOBAL_REBUILD_HPP

#include <cstddef> // `std::size_t`
#include <cstdio>  // `std::rename`, `std::remove`
#include <memory>  // `std::unique_ptr`
#include <new>     // `std::nothrow`
#include <string>  // `std::string`
#include <utility> // `std::move`, `std::forward`
#include <vector>  // `std::vector`

#include <usearch/index.hpp>
#include <usearch/index_dense.hpp>

namespace unum {
namespace usearch {

/**
 *  @brief  Orchestrates an interruptible, non-blocking global rebuild of a
 *          dense index, persisting a freshly reconstructed copy to disk.
 *
 *  @tparam index_at   A dense index type, i.e. an `index_dense_gt<...>`. The
 *                     adapter is deliberately written against that concrete
 *                     API (`add` / `remove` / `search` / `get` / `fork` /
 *                     `export_keys` / resumable `save_to_stream`) rather than
 *                     a generic concept.
 *  @tparam scalar_at  Scalar type used to shuttle vectors from primary to
 *                     shadow during migration. Defaults to 32-bit `float`.
 */
template <typename index_at, typename scalar_at = float> //
class global_rebuild_gt {
  public:
    using index_t = index_at;
    using scalar_t = scalar_at;
    using vector_key_t = typename index_t::vector_key_t;
    using add_result_t = typename index_t::add_result_t;
    using labeling_result_t = typename index_t::labeling_result_t;
    using search_result_t = typename index_t::search_result_t;

    /// @brief  Stage of the rebuild state machine.
    enum phase_t {
        phase_idle_k = 0,      ///< No rebuild in flight.
        phase_migrating_k = 1, ///< Re-inserting keys into the shadow index.
        phase_saving_k = 2,    ///< Streaming the frozen shadow to disk.
        phase_done_k = 3,      ///< Finished; file closed, tombstones replayed.
    };

    /// @brief  Boolean-convertible outcome, mirroring the index result types.
    struct result_t {
        error_t error{};
        explicit operator bool() const noexcept { return !error; }
        result_t failed(error_t message) noexcept {
            error = std::move(message);
            return std::move(*this);
        }
    };

  private:
    index_t* primary_ = nullptr;
    std::unique_ptr<index_t> shadow_;
    std::size_t budget_ = 256;
    phase_t phase_ = phase_idle_k;

    /// @brief  Key-set captured at `begin()` - the generation being rebuilt.
    std::vector<vector_key_t> migration_keys_;
    std::size_t migration_cursor_ = 0;

    /// @brief  Caller's destination path, and the `<path>.tmp` actually
    ///         written - renamed onto the destination only on completion.
    std::string final_path_;
    std::string temp_path_;
    output_file_t file_{nullptr};
    index_dense_serialized_state_t save_state_;

    /// @brief  Removals tombstoned while the rebuild is active.
    std::vector<vector_key_t> deferred_removes_;

  public:
    /**
     *  @param[in] primary      The live index to keep serving and to rebuild.
     *  @param[in] step_budget  Units of work per `step()`: vectors migrated, or
     *                          vectors/nodes serialized. Smaller budgets yield
     *                          shorter, more frequent pauses.
     */
    explicit global_rebuild_gt(index_t& primary, std::size_t step_budget = 256) noexcept
        : primary_(&primary), budget_(step_budget ? step_budget : 1) {}

    global_rebuild_gt(global_rebuild_gt const&) = delete;
    global_rebuild_gt& operator=(global_rebuild_gt const&) = delete;

    ~global_rebuild_gt() {
        // A rebuild abandoned before completion leaves a partial temp file;
        // the destination was never touched, so just discard the temp file.
        if (active()) {
            file_.close();
            std::remove(temp_path_.c_str());
        }
    }

    phase_t phase() const noexcept { return phase_; }
    bool active() const noexcept { return phase_ == phase_migrating_k || phase_ == phase_saving_k; }
    bool finished() const noexcept { return phase_ == phase_done_k; }
    std::size_t deferred_remove_count() const noexcept { return deferred_removes_.size(); }
    /// @brief  The reconstructed index, populated during `phase_migrating_k`
    ///         and `phase_saving_k`; released (null) at `phase_done_k`. Its
    ///         vectors alias the primary's storage - do not outlive it.
    index_t const* shadow() const noexcept { return shadow_.get(); }

    /// @brief  Insert a vector. Always routed to the primary, never blocked.
    template <typename scalar_other_at>
    add_result_t add(vector_key_t key, scalar_other_at const* vector) {
        return primary_->add(key, vector);
    }

    /**
     *  @brief  Remove a key. While a rebuild is active the physical removal is
     *          tombstoned and replayed on the primary once the rebuild ends,
     *          so the on-disk snapshot stays equal to the `begin()` generation.
     */
    labeling_result_t remove(vector_key_t key) {
        if (!active())
            return primary_->remove(key);
        deferred_removes_.push_back(key);
        labeling_result_t result;
        result.completed = 1;
        return result;
    }

    /// @brief  Nearest-neighbor search. Always serviced by the primary.
    template <typename scalar_other_at>
    search_result_t search(scalar_other_at const* query, std::size_t wanted) const {
        return primary_->search(query, wanted);
    }

    /// @brief  Fetch a stored vector by key. Always serviced by the primary.
    template <typename scalar_other_at>
    std::size_t get(vector_key_t key, scalar_other_at* vector, std::size_t count = 1) const {
        return primary_->get(key, vector, count);
    }

    bool contains(vector_key_t key) const { return primary_->contains(key); }
    std::size_t size() const noexcept { return primary_->size(); }

    /**
     *  @brief  Begin a global rebuild, persisting the result to @p path.
     *  @return A falsy ::result_t carrying an error message on failure.
     */
    result_t begin(char const* path) {
        result_t result;
        if (active())
            return result.failed("A global rebuild is already in flight");

        // The zero-copy migration reinterprets the primary's stored vector
        // bytes as `scalar_t`, so the adapter's scalar type must match the
        // index's native storage layout. Reject a mismatch up front rather
        // than silently corrupting the shadow.
        if (primary_->scalar_kind() != unum::usearch::scalar_kind<scalar_t>())
            return result.failed("Adapter scalar type must match the index's stored scalar kind");

        // Snapshot the live key-set: this exact generation is what we rebuild.
        std::size_t live = primary_->size();
        migration_keys_.resize(live);
        if (live)
            primary_->export_keys(migration_keys_.data(), 0, live);
        migration_cursor_ = 0;

        // A fresh, empty peer with the same metric and config - the shadow we
        // reconstruct the HNSW graph into from scratch, one re-insertion at a
        // time.
        typename index_t::copy_result_t forked = primary_->fork();
        if (!forked)
            return result.failed(std::move(forked.error));
        shadow_.reset(new (std::nothrow) index_t(std::move(forked.index)));
        if (!shadow_)
            return result.failed("Out of memory for the shadow index");
        if (live && !shadow_->try_reserve(live))
            return result.failed("Failed to reserve the shadow index");

        // Stream into a temp file; the destination keeps the previous index
        // until the completed file is atomically renamed into place.
        final_path_ = path;
        temp_path_ = final_path_ + ".tmp";
        file_ = output_file_t(temp_path_.c_str());
        serialization_result_t io = file_.open_if_not();
        if (!io)
            return result.failed(std::move(io.error));

        save_state_ = index_dense_serialized_state_t{};
        deferred_removes_.clear();
        phase_ = phase_migrating_k;
        return result;
    }

    /**
     *  @brief  Advance the rebuild by one budgeted chunk of work.
     *
     *  Does nothing once the rebuild is idle or finished. On the step that
     *  completes the save it closes the file and replays tombstoned removals.
     *
     *  @return A falsy ::result_t on error; otherwise truthy. Inspect `phase()`
     *          or `finished()` to learn whether more steps remain.
     */
    result_t step() {
        result_t result;

        // Stage A: migrate one budget's worth of keys into the shadow.
        if (phase_ == phase_migrating_k) {
            std::size_t migrated = 0;
            while (migrated < budget_ && migration_cursor_ < migration_keys_.size()) {
                vector_key_t key = migration_keys_[migration_cursor_++];
                byte_t const* vector = primary_->vector_data(key);
                // Removals are deferred, so a snapshot key should still be
                // present; tolerate a miss rather than abort the rebuild.
                if (!vector)
                    continue;
                // Zero-copy: the shadow node references the primary's stored
                // vector bytes (`copy_vector = false`) instead of duplicating
                // them, so only the graph is rebuilt, not the vectors.
                add_result_t added = shadow_->add(key, reinterpret_cast<scalar_t const*>(vector),
                                                  index_t::any_thread(), /*copy_vector=*/false);
                if (!added)
                    return result.failed(std::move(added.error));
                ++migrated;
            }
            if (migration_cursor_ >= migration_keys_.size())
                phase_ = phase_saving_k;
            return result;
        }

        // Stage B: stream one budget's worth of the frozen shadow to disk.
        if (phase_ == phase_saving_k) {
            serialization_result_t io;
            serialization_result_t saved = shadow_->save_to_stream(
                [&](void const* buffer, std::size_t length) {
                    io = file_.write(buffer, length);
                    return !!io;
                },
                save_state_, budget_);
            if (!saved)
                return result.failed(std::move(saved.error));
            if (save_state_.done()) {
                file_.close();
                // Atomically publish the finished snapshot. On POSIX `rename`
                // replaces the destination in one step, so a crash anywhere
                // before this point leaves the previous file fully intact.
                // (Windows `rename` cannot overwrite - the fallback there has
                // a tiny non-atomic window.)
                if (std::rename(temp_path_.c_str(), final_path_.c_str()) != 0) {
                    std::remove(final_path_.c_str());
                    if (std::rename(temp_path_.c_str(), final_path_.c_str()) != 0)
                        return result.failed("Failed to publish the rebuilt index file");
                }
                // Replay the tombstoned removals on the live primary now that
                // the snapshot is safely on disk.
                for (std::size_t i = 0; i != deferred_removes_.size(); ++i)
                    primary_->remove(deferred_removes_[i]);
                // Release the shadow: its job (producing the file) is done, and
                // its nodes alias the primary's vectors, so it must not outlive
                // an unsupervised primary. This also returns the one-graph
                // overhead the rebuild was holding.
                shadow_.reset();
                phase_ = phase_done_k;
            }
            return result;
        }

        return result; // Idle or already done - nothing to advance.
    }

    /**
     *  @brief  Drive the rebuild to completion, stepping until `finished()`.
     *
     *  Provided for tests and simple callers. A non-blocking caller should
     *  instead interleave its own work with individual `step()` calls.
     */
    result_t run_to_completion() {
        result_t result;
        while (active()) {
            result = step();
            if (!result)
                return result;
        }
        return result;
    }

};

} // namespace usearch
} // namespace unum

#endif // UNUM_USEARCH_GLOBAL_REBUILD_HPP
