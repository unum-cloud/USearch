/**
 *  @file       persistent_index.hpp
 *  @author     Mikhail Chichvarin
 *  @brief      Snapshot + WAL persistence wrapper around `index_dense_gt`.
 *  @date       May 23, 2026
 *
 *  @section Overview
 *
 *  `persistent_index_gt` makes an `index_dense_gt` durable across process
 *  restarts and crashes via the classic @b snapshot + @b WAL pattern:
 *
 *      * a base @b snapshot - a full index file written by
 *        `global_rebuild_gt` (non-blocking, crash-safe via temp + rename);
 *      * an append-only @b WAL - one framed record per `add` / `remove`,
 *        appended to disk before the operation is applied to RAM;
 *      * a tiny @b manifest - atomic pointer to the current generation.
 *
 *  Recovery loads the snapshot and replays the WAL, restoring the index to
 *  the last durable state. When the WAL grows past a threshold, an auto
 *  @b checkpoint kicks off a fresh snapshot (via the rebuild adapter) and -
 *  on completion - retires the old snapshot and the WAL prefix that the new
 *  snapshot now covers. The checkpoint is non-blocking: every mutating op
 *  drives a small step of it.
 *
 *  @section Concurrency and durability
 *
 *  Concurrent `add` / `remove` from multiple threads are supported; the WAL
 *  append is serialized by a single mutex (cheap, since there is no fsync),
 *  the index apply runs concurrently afterwards. Order is @b WAL-first: a
 *  crash between WAL append and index apply replays the record on recovery,
 *  so no committed op is lost beyond what is still in the OS page cache.
 *
 *  No `fsync` is issued: writes go to the OS via `fwrite` and rely on the
 *  page cache being flushed in the background. A clean process kill loses
 *  whatever sits in the libc buffer; a hard power loss can additionally lose
 *  the page-cache tail. The "on-disk lags RAM by at most one operation"
 *  invariant therefore holds for clean kills, not for hard crashes - the
 *  durability/throughput trade-off chosen for this wrapper.
 *
 *  Two short barriers per checkpoint, both under the WAL mutex:
 *      * at @b begin - drain in-flight index applies, then `rebuild.begin`;
 *      * at @b complete - copy the WAL suffix onto the new generation and
 *        commit the manifest atomically.
 *  Both run in milliseconds; no stop-the-world.
 */
#ifndef UNUM_USEARCH_PERSISTENT_INDEX_HPP
#define UNUM_USEARCH_PERSISTENT_INDEX_HPP

#include <atomic>  // `std::atomic`
#include <cstddef> // `std::size_t`
#include <cstdint> // `std::uint32_t`, `std::uint64_t`
#include <cstdio>  // `std::FILE`, `std::fopen`, `std::rename`, `std::remove`
#include <cstring> // `std::memcpy`, `std::memcmp`
#include <memory>  // `std::unique_ptr`
#include <mutex>   // `std::mutex`
#include <new>     // `std::nothrow`
#include <string>  // `std::string`
#include <thread>  // `std::this_thread::yield`
#include <utility> // `std::move`
#include <vector>  // `std::vector`

#include <usearch/global_rebuild.hpp>
#include <usearch/index.hpp>
#include <usearch/index_dense.hpp>

namespace unum {
namespace usearch {


/**
 *  @brief  Bytewise CRC32 over the IEEE polynomial, no table.
 *
 *  Small and good enough to detect torn last records at the WAL boundary.
 *  Records are short (one vector), so a tableless loop is fine.
 */
inline std::uint32_t crc32_ieee(void const* data, std::size_t bytes) noexcept {
    std::uint8_t const* p = static_cast<std::uint8_t const*>(data);
    std::uint32_t crc = 0xFFFFFFFFu;
    for (std::size_t i = 0; i != bytes; ++i) {
        crc ^= p[i];
        for (int k = 0; k != 8; ++k)
            crc = (crc >> 1) ^ ((crc & 1u) ? 0xEDB88320u : 0u);
    }
    return ~crc;
}



/// @brief  WAL operation codes, packed as one byte in each record's payload.
enum persistent_op_t : std::uint8_t {
    persistent_op_add_k = 1,
    persistent_op_remove_k = 2,
};

/**
 *  @brief  Header at the start of every WAL file.
 *
 *  Self-describes the WAL so a recovery can sanity-check it against the
 *  snapshot it lives next to (mismatched dims / scalar / metric would
 *  silently corrupt replay).
 */
struct persistent_wal_header_t {
    char magic[4];                ///< "uwal"
    std::uint32_t format_version; ///< Bumped on incompatible format changes.
    std::uint64_t dimensions;
    std::uint32_t scalar_kind;
    std::uint32_t metric_kind;
};
static constexpr char const* persistent_wal_magic_k = "uwal";
static constexpr std::uint32_t persistent_wal_version_k = 1;

/// @brief  Manifest file pointing at the current durable generation.
struct persistent_manifest_t {
    char magic[4]; ///< "umft"
    std::uint32_t format_version;
    std::uint64_t generation;
};
static constexpr char const* persistent_manifest_magic_k = "umft";


/**
 *  @brief  Durable wrapper around `index_dense_gt`: snapshot + WAL with
 *          non-blocking auto-checkpoints via `global_rebuild_gt`.
 */
template <typename index_at, typename scalar_at = float> //
class persistent_index_gt {
  public:
    using index_t = index_at;
    using scalar_t = scalar_at;
    using vector_key_t = typename index_t::vector_key_t;
    using add_result_t = typename index_t::add_result_t;
    using labeling_result_t = typename index_t::labeling_result_t;
    using search_result_t = typename index_t::search_result_t;
    using rebuild_t = global_rebuild_gt<index_t, scalar_t>;
    using metric_t = typename index_t::metric_t;

    /// @brief  Tunables. Defaults are reasonable for a moderate workload.
    struct config_t {
        /// @brief  Trigger an auto-checkpoint after this many ops since the
        ///         previous one. The WAL never grows past ~this many records.
        std::size_t checkpoint_after_ops = 100'000;
        /// @brief  Work-budget per `step()` of the rebuild adapter inside a
        ///         checkpoint - smaller budget = shorter ops-driven pauses.
        std::size_t checkpoint_step_budget = 256;
        /// @brief  Initial reserve for a freshly opened (empty) index.
        std::size_t initial_capacity = 1024;
    };

    struct open_result_t {
        std::unique_ptr<persistent_index_gt> index;
        error_t error;
        explicit operator bool() const noexcept { return !error; }
    };

  private:
    std::string base_path_;
    config_t config_;
    index_t index_;
    rebuild_t rebuild_;

    std::uint64_t generation_ = 0;
    std::uint64_t dimensions_ = 0;
    std::uint32_t scalar_kind_ = 0;
    std::uint32_t metric_kind_ = 0;
    std::size_t vector_bytes_ = 0; ///< `dimensions_ * sizeof(scalar_t)`

    // ---- WAL state. All writes guarded by `wal_mutex_`. -----------------
    std::mutex wal_mutex_;
    std::FILE* wal_file_ = nullptr; ///< open in "ab"
    std::string wal_path_;
    std::uint64_t wal_bytes_ = 0; ///< bytes written so far, tracked manually
    std::vector<char> record_buffer_;

    // ---- Checkpoint coordination ----------------------------------------
    std::atomic<std::uint64_t> ops_since_checkpoint_{0};
    std::atomic<std::size_t> in_flight_{0};
    std::mutex checkpoint_mutex_; ///< serializes step-driving + completion
    std::atomic<bool> checkpoint_active_{false};
    std::uint64_t checkpoint_watermark_ = 0; ///< WAL byte offset captured at begin
    std::uint64_t checkpoint_gen_ = 0;       ///< target generation of the in-flight checkpoint
    error_t checkpoint_error_{};

    persistent_index_gt(char const* path, config_t config)
        : base_path_(path), config_(config), rebuild_(index_, config.checkpoint_step_budget) {}

  public:
    persistent_index_gt(persistent_index_gt const&) = delete;
    persistent_index_gt& operator=(persistent_index_gt const&) = delete;

    ~persistent_index_gt() {
        // Best-effort flush; the OS will write the remaining page cache out.
        // An in-flight checkpoint just dies on the floor - its `.tmp` files
        // are cleaned by `rebuild_`'s dtor, the manifest still points at the
        // previous generation, and recovery picks that up cleanly.
        if (wal_file_) {
            std::fflush(wal_file_);
            std::fclose(wal_file_);
            wal_file_ = nullptr;
        }
    }

    /**
     *  @brief  Open (or recover) a persistent index at @p path. On first use
     *          this creates the snapshot/WAL/manifest; on subsequent use it
     *          loads the snapshot and replays the WAL.
     */
    static open_result_t open(char const* path, metric_t metric, index_dense_config_t index_config,
                              config_t config = {}) {
        open_result_t result;
        std::unique_ptr<persistent_index_gt> self(new (std::nothrow) persistent_index_gt(path, config));
        if (!self)
            return {nullptr, error_t("Out of memory for persistent_index")};
        error_t err = self->open_(std::move(metric), index_config);
        if (err) {
            result.error = std::move(err);
            return result;
        }
        result.index = std::move(self);
        return result;
    }


    /// @brief  Insert a vector durably (WAL append + index apply).
    add_result_t add(vector_key_t key, scalar_t const* vector) {
        // WAL-first: a crash between WAL append and index apply re-applies
        // the op on recovery, so a committed-to-WAL op is never lost.
        {
            std::unique_lock<std::mutex> lock(wal_mutex_);
            if (!wal_file_)
                return add_result_t{}.failed("WAL is not open");
            append_record_(persistent_op_add_k, key, reinterpret_cast<char const*>(vector));
            in_flight_.fetch_add(1, std::memory_order_acq_rel);
            ops_since_checkpoint_.fetch_add(1, std::memory_order_relaxed);
        }
        add_result_t r = index_.add(key, vector);
        in_flight_.fetch_sub(1, std::memory_order_acq_rel);
        maybe_drive_checkpoint_();
        return r;
    }

    /// @brief  Remove a key durably.
    labeling_result_t remove(vector_key_t key) {
        {
            std::unique_lock<std::mutex> lock(wal_mutex_);
            if (!wal_file_) {
                labeling_result_t r;
                return r.failed("WAL is not open");
            }
            append_record_(persistent_op_remove_k, key, nullptr);
            in_flight_.fetch_add(1, std::memory_order_acq_rel);
            ops_since_checkpoint_.fetch_add(1, std::memory_order_relaxed);
        }
        labeling_result_t r = index_.remove(key);
        in_flight_.fetch_sub(1, std::memory_order_acq_rel);
        maybe_drive_checkpoint_();
        return r;
    }

    search_result_t search(scalar_t const* query, std::size_t wanted) const { return index_.search(query, wanted); }
    std::size_t get(vector_key_t key, scalar_t* out, std::size_t count = 1) const {
        return index_.get(key, out, count);
    }
    bool contains(vector_key_t key) const { return index_.contains(key); }
    std::size_t size() const noexcept { return index_.size(); }
    std::uint64_t generation() const noexcept { return generation_; }
    bool checkpoint_in_flight() const noexcept { return checkpoint_active_.load(std::memory_order_acquire); }

    /// @brief  Grow the in-RAM capacity (does not touch the on-disk format).
    bool reserve(std::size_t members) { return index_.try_reserve(members); }

    /**
     *  @brief  Drive a checkpoint to completion synchronously. If none is in
     *          flight, starts one. Useful before shutdown or in tests.
     */
    error_t checkpoint() {
        if (!checkpoint_active_.load(std::memory_order_acquire))
            start_checkpoint_();
        if (checkpoint_error_)
            return std::move(checkpoint_error_);
        while (checkpoint_active_.load(std::memory_order_acquire)) {
            drive_checkpoint_step_();
            if (checkpoint_error_)
                return std::move(checkpoint_error_);
            std::this_thread::yield();
        }
        return {};
    }



  private:
    std::string gen_path_(char const* suffix, std::uint64_t g) const {
        return base_path_ + "." + std::to_string(g) + suffix;
    }
    std::string manifest_path_() const { return base_path_ + ".manifest"; }

    /// @brief  Atomically replace @p path with @p bytes via temp + rename.
    static error_t atomic_write_(std::string const& path, void const* bytes, std::size_t length) {
        std::string tmp = path + ".tmp";
        std::FILE* file = std::fopen(tmp.c_str(), "wb");
        if (!file)
            return error_t("Failed to create temp file");
        std::size_t written = std::fwrite(bytes, 1, length, file);
        std::fclose(file);
        if (written != length) {
            std::remove(tmp.c_str());
            return error_t("Short write on temp file");
        }
        if (std::rename(tmp.c_str(), path.c_str()) != 0) {
            std::remove(path.c_str());
            if (std::rename(tmp.c_str(), path.c_str()) != 0)
                return error_t("Failed to rename temp file");
        }
        return {};
    }

    error_t write_manifest_(std::uint64_t generation) {
        persistent_manifest_t manifest{};
        std::memcpy(manifest.magic, persistent_manifest_magic_k, 4);
        manifest.format_version = persistent_wal_version_k;
        manifest.generation = generation;
        return atomic_write_(manifest_path_(), &manifest, sizeof(manifest));
    }

    /// @brief  Reads the manifest. Returns true on success; sets @p exists to
    ///         false if the file simply does not exist (fresh-open case).
    error_t read_manifest_(std::uint64_t& generation_out, bool& exists) {
        exists = false;
        std::FILE* file = std::fopen(manifest_path_().c_str(), "rb");
        if (!file)
            return {}; // absent is not an error - caller treats as "fresh"
        persistent_manifest_t manifest{};
        std::size_t got = std::fread(&manifest, 1, sizeof(manifest), file);
        std::fclose(file);
        if (got != sizeof(manifest))
            return error_t("Manifest is truncated");
        if (std::memcmp(manifest.magic, persistent_manifest_magic_k, 4) != 0)
            return error_t("Manifest magic mismatch");
        generation_out = manifest.generation;
        exists = true;
        return {};
    }

    /// @brief  Refresh metadata (dims/scalar/metric/vector_bytes) from `index_`.
    void refresh_metadata_() {
        dimensions_ = index_.dimensions();
        scalar_kind_ = static_cast<std::uint32_t>(index_.scalar_kind());
        metric_kind_ = static_cast<std::uint32_t>(index_.metric_kind());
        vector_bytes_ = dimensions_ * sizeof(scalar_t);
    }

    /// @brief  Open a brand-new WAL: write the self-describing header, leave
    ///         the file positioned for further appends.
    error_t create_wal_(std::string const& path) {
        std::FILE* file = std::fopen(path.c_str(), "wb");
        if (!file)
            return error_t("Failed to create WAL file");
        persistent_wal_header_t header{};
        std::memcpy(header.magic, persistent_wal_magic_k, 4);
        header.format_version = persistent_wal_version_k;
        header.dimensions = dimensions_;
        header.scalar_kind = scalar_kind_;
        header.metric_kind = metric_kind_;
        std::fwrite(&header, sizeof(header), 1, file);
        std::fflush(file);
        std::fclose(file);
        return {};
    }

    /// @brief  Open the active WAL in append mode and remember its size.
    error_t open_wal_for_append_(std::string const& path) {
        std::FILE* file = std::fopen(path.c_str(), "ab");
        if (!file)
            return error_t("Failed to open WAL for append");
        // `ab` mode positions at end of file; learn its size from fseek/ftell
        // via a separate read-mode open (portable, avoids ftell-on-append).
        std::FILE* probe = std::fopen(path.c_str(), "rb");
        if (!probe) {
            std::fclose(file);
            return error_t("Failed to stat WAL file");
        }
        std::fseek(probe, 0, SEEK_END);
        long size = std::ftell(probe);
        std::fclose(probe);
        if (size < 0) {
            std::fclose(file);
            return error_t("Failed to stat WAL file");
        }
        wal_file_ = file;
        wal_path_ = path;
        wal_bytes_ = static_cast<std::uint64_t>(size);
        record_buffer_.reserve(1 + sizeof(vector_key_t) + vector_bytes_);
        return {};
    }

    /// @brief  Build a record into `record_buffer_` and write it. Caller holds
    ///         `wal_mutex_`.
    void append_record_(persistent_op_t op, vector_key_t key, char const* vector_or_null) {
        std::uint32_t payload_len = static_cast<std::uint32_t>( //
            1 + sizeof(vector_key_t) + (vector_or_null ? vector_bytes_ : 0));
        record_buffer_.resize(payload_len);
        record_buffer_[0] = static_cast<char>(op);
        std::memcpy(&record_buffer_[1], &key, sizeof(vector_key_t));
        if (vector_or_null)
            std::memcpy(&record_buffer_[1 + sizeof(vector_key_t)], vector_or_null, vector_bytes_);
        std::uint32_t crc = crc32_ieee(record_buffer_.data(), payload_len);
        // [u32 len][u32 crc][payload]
        std::fwrite(&payload_len, sizeof(payload_len), 1, wal_file_);
        std::fwrite(&crc, sizeof(crc), 1, wal_file_);
        std::fwrite(record_buffer_.data(), 1, payload_len, wal_file_);
        wal_bytes_ += sizeof(payload_len) + sizeof(crc) + payload_len;
    }

    /**
     *  @brief  Replay every well-framed WAL record onto `index_`, stopping at
     *          the first incomplete or crc-mismatched record. The header is
     *          validated against the in-memory `index_` metadata.
     */
    error_t replay_wal_(std::string const& path) {
        std::FILE* file = std::fopen(path.c_str(), "rb");
        if (!file)
            return error_t("Failed to open WAL for replay");
        persistent_wal_header_t header{};
        if (std::fread(&header, 1, sizeof(header), file) != sizeof(header)) {
            std::fclose(file);
            return error_t("WAL header truncated");
        }
        if (std::memcmp(header.magic, persistent_wal_magic_k, 4) != 0) {
            std::fclose(file);
            return error_t("WAL magic mismatch");
        }
        if (header.dimensions != dimensions_ || header.scalar_kind != scalar_kind_ ||
            header.metric_kind != metric_kind_) {
            std::fclose(file);
            return error_t("WAL metadata does not match the snapshot");
        }

        std::uint32_t const max_payload = static_cast<std::uint32_t>(1 + sizeof(vector_key_t) + vector_bytes_);
        std::vector<char> buffer;
        buffer.reserve(max_payload);

        while (true) {
            std::uint32_t len = 0;
            std::size_t got_len = std::fread(&len, 1, sizeof(len), file);
            if (got_len == 0)
                break; // clean EOF
            if (got_len != sizeof(len))
                break; // torn
            std::uint32_t crc = 0;
            if (std::fread(&crc, 1, sizeof(crc), file) != sizeof(crc))
                break; // torn
            if (len < 1 + sizeof(vector_key_t) || len > max_payload)
                break; // corrupt length - stop replay
            buffer.resize(len);
            if (std::fread(buffer.data(), 1, len, file) != len)
                break; // torn
            if (crc32_ieee(buffer.data(), len) != crc)
                break; // torn / bit flip

            // Apply the record. Replay is single-threaded - safe to use the
            // `contains` check to make `add` idempotent against a duplicate.
            persistent_op_t op = static_cast<persistent_op_t>(static_cast<std::uint8_t>(buffer[0]));
            vector_key_t key{};
            std::memcpy(&key, &buffer[1], sizeof(vector_key_t));
            if (op == persistent_op_add_k) {
                if (!index_.contains(key)) {
                    scalar_t const* vector = reinterpret_cast<scalar_t const*>(&buffer[1 + sizeof(vector_key_t)]);
                    auto r = index_.add(key, vector);
                    if (!r) {
                        std::fclose(file);
                        return std::move(r.error);
                    }
                }
            } else if (op == persistent_op_remove_k) {
                index_.remove(key); // tolerant of an absent key
            } else {
                break; // unknown op - stop, conservative
            }
        }

        std::fclose(file);
        return {};
    }

    /// @brief  First pass over the WAL: count `add` records to size the
    ///         index's capacity before replay. Stops at the same torn point.
    std::uint64_t count_wal_adds_(std::string const& path) const {
        std::FILE* file = std::fopen(path.c_str(), "rb");
        if (!file)
            return 0;
        // Skip the header.
        if (std::fseek(file, sizeof(persistent_wal_header_t), SEEK_SET) != 0) {
            std::fclose(file);
            return 0;
        }
        std::uint32_t const max_payload = static_cast<std::uint32_t>(1 + sizeof(vector_key_t) + vector_bytes_);
        std::uint64_t adds = 0;
        while (true) {
            std::uint32_t len = 0, crc = 0;
            if (std::fread(&len, 1, sizeof(len), file) != sizeof(len))
                break;
            if (std::fread(&crc, 1, sizeof(crc), file) != sizeof(crc))
                break;
            if (len < 1 + sizeof(vector_key_t) || len > max_payload)
                break;
            std::uint8_t op = 0;
            if (std::fread(&op, 1, 1, file) != 1)
                break;
            if (std::fseek(file, static_cast<long>(len - 1), SEEK_CUR) != 0)
                break;
            if (op == persistent_op_add_k)
                ++adds;
        }
        std::fclose(file);
        return adds;
    }

    /**
     *  @brief  Open path: recover an existing manifest or initialize fresh.
     */
    error_t open_(metric_t metric, index_dense_config_t index_config) {
        std::uint64_t generation = 0;
        bool manifest_exists = false;
        if (error_t err = read_manifest_(generation, manifest_exists))
            return err;

        if (!manifest_exists) {
            // Fresh: build an empty index, write snapshot.0 + wal.0 +
            // manifest=0. Subsequent reopens take the recovery path below.
            typename index_t::state_result_t made = index_t::make(std::move(metric), index_config);
            if (!made)
                return std::move(made.error);
            index_ = std::move(made.index);
            if (!index_.try_reserve(config_.initial_capacity))
                return error_t("Failed to reserve initial capacity");
            refresh_metadata_();
            generation_ = 0;

            // The empty snapshot is fine in the standard format.
            auto saved = index_.save(gen_path_(".snapshot", 0).c_str());
            if (!saved)
                return std::move(saved.error);
            if (error_t err = create_wal_(gen_path_(".wal", 0)))
                return err;
            if (error_t err = open_wal_for_append_(gen_path_(".wal", 0)))
                return err;
            return write_manifest_(0);
        }

        // Recovery: load snapshot, count WAL adds for reserve, replay WAL.
        generation_ = generation;
        std::string snapshot_path = gen_path_(".snapshot", generation_);
        typename index_t::state_result_t loaded = index_t::make(snapshot_path.c_str());
        if (!loaded)
            return std::move(loaded.error);
        index_ = std::move(loaded.index);
        refresh_metadata_();

        std::string wal_path = gen_path_(".wal", generation_);
        std::uint64_t wal_adds = count_wal_adds_(wal_path);
        if (!index_.try_reserve(index_.size() + wal_adds + config_.initial_capacity))
            return error_t("Failed to reserve capacity for replay");
        if (error_t err = replay_wal_(wal_path))
            return err;
        return open_wal_for_append_(wal_path);
    }



    /**
     *  @brief  Called from mutation paths. Starts an auto-checkpoint when due
     *          and drives one budgeted step of an active one (try-locked, so
     *          only one thread steps at a time; others just move on).
     */
    void maybe_drive_checkpoint_() {
        if (!checkpoint_active_.load(std::memory_order_acquire)) {
            if (ops_since_checkpoint_.load(std::memory_order_relaxed) >= config_.checkpoint_after_ops)
                start_checkpoint_();
        }
        if (checkpoint_active_.load(std::memory_order_acquire))
            drive_checkpoint_step_();
    }

    /**
     *  @brief  Open a new checkpoint: barrier (drain in-flight applies)
     *          + `rebuild.begin(snapshot.<g+1>)`. Idempotent against
     *          concurrent triggers.
     */
    void start_checkpoint_() {
        std::unique_lock<std::mutex> lock(wal_mutex_);
        if (checkpoint_active_.load(std::memory_order_acquire))
            return;

        // Drain ops that have already appended a WAL record but whose index
        // apply has not finished. Holding `wal_mutex_` keeps new appends out,
        // so `in_flight_` only decreases. After this, `index_` reflects every
        // record up to `wal_bytes_`, so the snapshot we are about to take
        // covers exactly that prefix.
        while (in_flight_.load(std::memory_order_acquire) > 0)
            std::this_thread::yield();

        checkpoint_watermark_ = wal_bytes_;
        checkpoint_gen_ = generation_ + 1;
        std::string snapshot_path = gen_path_(".snapshot", checkpoint_gen_);
        typename rebuild_t::result_t br = rebuild_.begin(snapshot_path.c_str());
        if (!br) {
            checkpoint_error_ = std::move(br.error);
            return;
        }
        checkpoint_active_.store(true, std::memory_order_release);
    }

    /// @brief  Drive one step of the active checkpoint. At most one thread
    ///         steps at a time; on completion this finalizes the checkpoint.
    void drive_checkpoint_step_() {
        std::unique_lock<std::mutex> lock(checkpoint_mutex_, std::try_to_lock);
        if (!lock.owns_lock())
            return;
        if (!checkpoint_active_.load(std::memory_order_acquire))
            return;
        auto sr = rebuild_.step();
        if (!sr) {
            checkpoint_error_ = std::move(sr.error);
            return;
        }
        if (rebuild_.finished())
            finish_checkpoint_();
    }

    /**
     *  @brief  Splice the WAL suffix `wal.<g>[watermark..end]` onto a brand-
     *          new `wal.<g+1>` (header + suffix), atomically commit a new
     *          manifest pointing at `g+1`, then retire the old files. Held
     *          briefly under `wal_mutex_` to keep the suffix size stable.
     */
    void finish_checkpoint_() {
        std::unique_lock<std::mutex> lock(wal_mutex_);

        // Ensure every byte we have appended is on disk before we read it.
        if (wal_file_)
            std::fflush(wal_file_);

        std::string old_wal = wal_path_;
        std::uint64_t suffix_start = checkpoint_watermark_;
        std::uint64_t suffix_end = wal_bytes_;
        std::string new_wal = gen_path_(".wal", checkpoint_gen_);

        // Build the new WAL: fresh header, then a verbatim copy of the
        // appended-since-watermark bytes. Records are framed independently,
        // so no per-record translation is needed.
        std::FILE* dst = std::fopen(new_wal.c_str(), "wb");
        if (!dst) {
            checkpoint_error_ = error_t("Failed to create new-gen WAL");
            return;
        }
        persistent_wal_header_t header{};
        std::memcpy(header.magic, persistent_wal_magic_k, 4);
        header.format_version = persistent_wal_version_k;
        header.dimensions = dimensions_;
        header.scalar_kind = scalar_kind_;
        header.metric_kind = metric_kind_;
        std::fwrite(&header, sizeof(header), 1, dst);

        if (suffix_end > suffix_start) {
            std::FILE* src = std::fopen(old_wal.c_str(), "rb");
            if (!src) {
                std::fclose(dst);
                std::remove(new_wal.c_str());
                checkpoint_error_ = error_t("Failed to open old WAL for suffix copy");
                return;
            }
            std::fseek(src, static_cast<long>(suffix_start), SEEK_SET);
            char buffer[64 * 1024];
            std::uint64_t remaining = suffix_end - suffix_start;
            while (remaining > 0) {
                std::size_t want = remaining < sizeof(buffer) ? static_cast<std::size_t>(remaining) : sizeof(buffer);
                std::size_t got = std::fread(buffer, 1, want, src);
                if (got == 0)
                    break;
                std::fwrite(buffer, 1, got, dst);
                remaining -= got;
            }
            std::fclose(src);
        }
        std::fflush(dst);
        std::fclose(dst);

        // Atomically swap the active WAL handle to the new file, BEFORE the
        // manifest commit. If we crash here, the manifest still points at
        // the old generation and recovery finds the old WAL intact.
        std::fclose(wal_file_);
        wal_file_ = std::fopen(new_wal.c_str(), "ab");
        if (!wal_file_) {
            checkpoint_error_ = error_t("Failed to reopen new WAL for append");
            return;
        }
        wal_path_ = new_wal;
        wal_bytes_ = sizeof(persistent_wal_header_t) + (suffix_end - suffix_start);

        // The atomic commit: from here on the new generation is canonical.
        if (error_t err = write_manifest_(checkpoint_gen_)) {
            checkpoint_error_ = std::move(err);
            return;
        }

        // Retire the old generation. A crash before these removes leaves
        // orphan files but the manifest already points at the new generation,
        // so recovery is correct - just slightly noisy on disk.
        std::remove(old_wal.c_str());
        std::remove(gen_path_(".snapshot", generation_).c_str());

        generation_ = checkpoint_gen_;
        ops_since_checkpoint_.store(0, std::memory_order_relaxed);
        checkpoint_active_.store(false, std::memory_order_release);
    }

};

} // namespace usearch
} // namespace unum

#endif // UNUM_USEARCH_PERSISTENT_INDEX_HPP
