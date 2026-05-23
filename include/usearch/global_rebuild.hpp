/**
 *  @file       global_rebuild.hpp
 *  @author     Mikhail Chichvarin
 *  @brief      Non-blocking @b global-rebuild orchestrator for `index_dense_gt`.
 *
 *  Persists the HNSW index to disk without a stop-the-world `save`. Rebuilds
 *  it into a fresh "shadow" peer by re-insertion, then streams that frozen
 *  shadow to disk through the resumable `save_to_stream`. Both phases run in
 *  budgeted steps; the live "primary" keeps serving reads and writes.
 *
 *      1. `phase_migrating` - re-insert the key-set captured at `begin()`
 *         into the shadow, with `copy_vector = false` so shadow nodes alias
 *         the primary's vector bytes (extra RAM during a rebuild is one
 *         graph, not a full clone).
 *      2. `phase_saving` - stream the now-frozen shadow into `<path>.tmp`.
 *      3. `phase_done` - atomic `rename` onto `<path>`, release the shadow,
 *         replay tombstoned removes.
 *
 *  Concurrent mutations: `add` always hits the primary; `remove` during a
 *  rebuild is tombstoned and replayed at completion, so the on-disk snapshot
 *  equals the `begin()` generation exactly.
 *
 *  Crash safety: writes go through `<path>.tmp` and become canonical only on
 *  the `rename` (atomic on POSIX). A crash at any point leaves either the
 *  previous file or the new file, never a truncated one. This is not resume-
 *  across-restart: the continuation cursor lives in RAM, so a killed rebuild
 *  is restarted from `begin`, not continued.
 */
#ifndef UNUM_USEARCH_GLOBAL_REBUILD_HPP
#define UNUM_USEARCH_GLOBAL_REBUILD_HPP

#include <cstddef>
#include <cstdio>
#include <memory>
#include <new>
#include <string>
#include <utility>
#include <vector>

#include <usearch/index.hpp>
#include <usearch/index_dense.hpp>

namespace unum {
namespace usearch {

/// @brief  Orchestrates an interruptible, non-blocking global rebuild of a
///         dense index, persisting a freshly reconstructed copy to disk.
///         `scalar_at` must match the index's stored scalar kind - the
///         migration reinterprets the primary's raw vector bytes as it.
template <typename index_at, typename scalar_at = float> //
class global_rebuild_gt {
  public:
    using index_t = index_at;
    using scalar_t = scalar_at;
    using vector_key_t = typename index_t::vector_key_t;
    using add_result_t = typename index_t::add_result_t;
    using labeling_result_t = typename index_t::labeling_result_t;
    using search_result_t = typename index_t::search_result_t;

    enum phase_t {
        phase_idle_k = 0,
        phase_migrating_k = 1,
        phase_saving_k = 2,
        phase_done_k = 3,
    };

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

    std::vector<vector_key_t> migration_keys_;
    std::size_t migration_cursor_ = 0;

    // Written into `<path>.tmp`, renamed onto `<path>` only on completion.
    std::string final_path_;
    std::string temp_path_;
    output_file_t file_{nullptr};
    index_dense_serialized_state_t save_state_;

    std::vector<vector_key_t> deferred_removes_;

  public:
    /// @param[in] step_budget  Vectors migrated, or vectors/nodes serialized,
    ///                         per `step()`. Smaller = shorter pauses.
    explicit global_rebuild_gt(index_t& primary, std::size_t step_budget = 256) noexcept
        : primary_(&primary), budget_(step_budget ? step_budget : 1) {}

    global_rebuild_gt(global_rebuild_gt const&) = delete;
    global_rebuild_gt& operator=(global_rebuild_gt const&) = delete;

    ~global_rebuild_gt() {
        // Abandoned mid-rebuild: the destination was never touched, so just
        // discard the temp file.
        if (active()) {
            file_.close();
            std::remove(temp_path_.c_str());
        }
    }

    phase_t phase() const noexcept { return phase_; }
    bool active() const noexcept { return phase_ == phase_migrating_k || phase_ == phase_saving_k; }
    bool finished() const noexcept { return phase_ == phase_done_k; }
    std::size_t deferred_remove_count() const noexcept { return deferred_removes_.size(); }
    /// @brief  Populated during the rebuild, released (null) at `phase_done_k`.
    ///         Vectors alias the primary's storage - do not outlive it.
    index_t const* shadow() const noexcept { return shadow_.get(); }

    /// @brief  Always routed to the primary, never blocked.
    template <typename scalar_other_at>
    add_result_t add(vector_key_t key, scalar_other_at const* vector) { return primary_->add(key, vector); }

    /// @brief  Tombstoned during an active rebuild, replayed at completion,
    ///         so the on-disk snapshot stays equal to the `begin()` generation.
    labeling_result_t remove(vector_key_t key) {
        if (!active())
            return primary_->remove(key);
        deferred_removes_.push_back(key);
        labeling_result_t result;
        result.completed = 1;
        return result;
    }

    /// @brief  Begin a global rebuild, persisting the result to @p path.
    result_t begin(char const* path) {
        result_t result;
        if (active())
            return result.failed("A global rebuild is already in flight");
        // Zero-copy migration reinterprets the primary's stored bytes as
        // `scalar_t`, so the adapter's scalar type must match.
        if (primary_->scalar_kind() != unum::usearch::scalar_kind<scalar_t>())
            return result.failed("Adapter scalar type must match the index's stored scalar kind");

        std::size_t live = primary_->size();
        migration_keys_.resize(live);
        if (live)
            primary_->export_keys(migration_keys_.data(), 0, live);
        migration_cursor_ = 0;

        typename index_t::copy_result_t forked = primary_->fork();
        if (!forked)
            return result.failed(std::move(forked.error));
        shadow_.reset(new (std::nothrow) index_t(std::move(forked.index)));
        if (!shadow_)
            return result.failed("Out of memory for the shadow index");
        if (live && !shadow_->try_reserve(live))
            return result.failed("Failed to reserve the shadow index");

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

    /// @brief  Advance the rebuild by one budgeted chunk. Idempotent when
    ///         idle or finished. Closes the file and replays tombstones on
    ///         the step that completes the save.
    result_t step() {
        result_t result;

        if (phase_ == phase_migrating_k) {
            std::size_t migrated = 0;
            while (migrated < budget_ && migration_cursor_ < migration_keys_.size()) {
                vector_key_t key = migration_keys_[migration_cursor_++];
                byte_t const* vector = primary_->vector_data(key);
                if (!vector)
                    continue; // key already gone (a deferred-remove race-with-self)
                // Zero-copy: shadow aliases primary's vector bytes.
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
                // Atomic publish on POSIX; Windows `rename` cannot overwrite,
                // hence the fallback with its tiny non-atomic window there.
                if (std::rename(temp_path_.c_str(), final_path_.c_str()) != 0) {
                    std::remove(final_path_.c_str());
                    if (std::rename(temp_path_.c_str(), final_path_.c_str()) != 0)
                        return result.failed("Failed to publish the rebuilt index file");
                }
                for (std::size_t i = 0; i != deferred_removes_.size(); ++i)
                    primary_->remove(deferred_removes_[i]);
                // Shadow nodes alias the primary's vectors - don't outlive it.
                shadow_.reset();
                phase_ = phase_done_k;
            }
            return result;
        }

        return result;
    }
};

} // namespace usearch
} // namespace unum

#endif // UNUM_USEARCH_GLOBAL_REBUILD_HPP
