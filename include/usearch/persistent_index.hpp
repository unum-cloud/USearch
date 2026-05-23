/**
 *  @file       persistent_index.hpp
 *  @author     Mikhail Chichvarin
 *  @brief      Snapshot + WAL persistence wrapper around `index_dense_gt`.
 *
 *  On-disk (base name `<path>`):
 *      <path>.<g>.snapshot - full index file (written by `global_rebuild_gt`,
 *                            crash-safe via temp + rename);
 *      <path>.<g>.wal      - append-only log, framed
 *                            `[u32 len][u32 crc32][u8 op][key][vec]`;
 *      <path>.manifest     - atomic pointer to the current generation `g`.
 *
 *  Order is WAL-first: append the record, then apply to RAM. A crash between
 *  the two replays the record on recovery, so committed ops are not lost
 *  beyond what is still in the OS page cache. No `fsync` - durability is
 *  best-effort vs hard crashes, traded for throughput.
 *
 *  Auto-checkpoint when ops since last cross `checkpoint_after_ops`:
 *      begin    - under `wal_mutex_`, drain in-flight index applies, snapshot
 *                 the WAL offset as a watermark, `rebuild.begin(snapshot.<g+1>)`;
 *      step     - each mutating op drives one budgeted `rebuild.step()`
 *                 (try-locked, so only one stepper at a time);
 *      complete - under `wal_mutex_`, splice `wal.<g>[watermark..end]` onto
 *                 a fresh `wal.<g+1>` (header + suffix), atomically commit
 *                 the manifest, retire the old snapshot and WAL.
 *  Both barriers are ~ms. A crash before the manifest commit leaves the
 *  previous generation canonical.
 */
#ifndef UNUM_USEARCH_PERSISTENT_INDEX_HPP
#define UNUM_USEARCH_PERSISTENT_INDEX_HPP

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <usearch/global_rebuild.hpp>
#include <usearch/index.hpp>
#include <usearch/index_dense.hpp>

namespace unum {
namespace usearch {

/// @brief  Bytewise CRC32 (IEEE polynomial). Records are short; no table.
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

enum persistent_op_t : std::uint8_t {
    persistent_op_add_k = 1,
    persistent_op_remove_k = 2,
};

/// @brief  RAII wrapper over a C `FILE*`: closes on scope exit, drops every
///         `std::fclose(file); return error_t(...)` paired cleanup.
struct file_closer_t {
    void operator()(std::FILE* f) const noexcept {
        if (f)
            std::fclose(f);
    }
};
using file_t = std::unique_ptr<std::FILE, file_closer_t>;

/// @brief  Self-describing header at the start of every WAL file.
struct persistent_wal_header_t {
    char magic[4]; // "uwal"
    std::uint32_t format_version;
    std::uint64_t dimensions;
    std::uint32_t scalar_kind;
    std::uint32_t metric_kind;
};
static constexpr char const* persistent_wal_magic_k = "uwal";
static constexpr std::uint32_t persistent_wal_version_k = 1;

/// @brief  Atomic pointer to the current durable generation.
struct persistent_manifest_t {
    char magic[4]; // "umft"
    std::uint32_t format_version;
    std::uint64_t generation;
};
static constexpr char const* persistent_manifest_magic_k = "umft";

/// @brief  Durable wrapper around `index_dense_gt`: snapshot + WAL with
///         non-blocking auto-checkpoints via `global_rebuild_gt`.
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

    struct config_t {
        std::size_t checkpoint_after_ops = 100'000;
        std::size_t checkpoint_step_budget = 256;
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
    std::size_t vector_bytes_ = 0;

    // WAL state. All file writes happen under `wal_mutex_`.
    std::mutex wal_mutex_;
    file_t wal_file_;
    std::string wal_path_;
    std::uint64_t wal_bytes_ = 0;
    std::vector<char> record_buffer_;

    std::atomic<std::uint64_t> ops_since_checkpoint_{0};
    std::atomic<std::size_t> in_flight_{0};
    std::mutex checkpoint_mutex_;
    std::atomic<bool> checkpoint_active_{false};
    std::uint64_t checkpoint_watermark_ = 0;
    std::uint64_t checkpoint_gen_ = 0;
    error_t checkpoint_error_{};

    persistent_index_gt(char const* path, config_t config)
        : base_path_(path), config_(config), rebuild_(index_, config.checkpoint_step_budget) {}

  public:
    persistent_index_gt(persistent_index_gt const&) = delete;
    persistent_index_gt& operator=(persistent_index_gt const&) = delete;

    ~persistent_index_gt() {
        // Best-effort flush; the OS will write the rest of the page cache out.
        // An in-flight checkpoint is just dropped - its temp file is cleaned
        // by `rebuild_`'s dtor; the manifest still points at the previous
        // generation and recovery is consistent. `wal_file_` closes itself.
        if (wal_file_)
            std::fflush(wal_file_.get());
    }

    /// @brief  Open (or recover) a persistent index at @p path.
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

    add_result_t add(vector_key_t key, scalar_t const* vector) {
        // WAL-first: a crash between WAL append and index apply replays the
        // record on recovery, so a committed-to-WAL op is never lost.
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
    /// @brief  Drive a checkpoint to completion (starts one if needed).
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

    /// @brief  Atomically replace the manifest via temp + rename. On POSIX
    ///         the rename is atomic; the Windows fallback has a tiny window.
    error_t write_manifest_(std::uint64_t generation) {
        persistent_manifest_t manifest{};
        std::memcpy(manifest.magic, persistent_manifest_magic_k, 4);
        manifest.format_version = persistent_wal_version_k;
        manifest.generation = generation;
        std::string path = manifest_path_();
        std::string tmp = path + ".tmp";
        std::size_t written;
        {
            file_t file{std::fopen(tmp.c_str(), "wb")};
            if (!file)
                return error_t("Failed to create manifest temp file");
            written = std::fwrite(&manifest, 1, sizeof(manifest), file.get());
        }
        if (written != sizeof(manifest)) {
            std::remove(tmp.c_str());
            return error_t("Short write on manifest temp file");
        }
        if (std::rename(tmp.c_str(), path.c_str()) != 0) {
            std::remove(path.c_str());
            if (std::rename(tmp.c_str(), path.c_str()) != 0)
                return error_t("Failed to rename manifest");
        }
        return {};
    }

    void refresh_metadata_() {
        dimensions_ = index_.dimensions();
        scalar_kind_ = static_cast<std::uint32_t>(index_.scalar_kind());
        metric_kind_ = static_cast<std::uint32_t>(index_.metric_kind());
        vector_bytes_ = dimensions_ * sizeof(scalar_t);
    }

    /// @brief  Build a WAL header reflecting current metadata.
    persistent_wal_header_t make_wal_header_() const {
        persistent_wal_header_t header{};
        std::memcpy(header.magic, persistent_wal_magic_k, 4);
        header.format_version = persistent_wal_version_k;
        header.dimensions = dimensions_;
        header.scalar_kind = scalar_kind_;
        header.metric_kind = metric_kind_;
        return header;
    }

    /// @brief  Open @p path in append mode, learning its current size.
    error_t open_wal_for_append_(std::string const& path) {
        file_t file{std::fopen(path.c_str(), "ab")};
        if (!file)
            return error_t("Failed to open WAL for append");
        long size = -1;
        if (file_t probe{std::fopen(path.c_str(), "rb")}) {
            std::fseek(probe.get(), 0, SEEK_END);
            size = std::ftell(probe.get());
        }
        if (size < 0)
            return error_t("Failed to stat WAL file");
        wal_file_ = std::move(file);
        wal_path_ = path;
        wal_bytes_ = static_cast<std::uint64_t>(size);
        record_buffer_.reserve(1 + sizeof(vector_key_t) + vector_bytes_);
        return {};
    }

    /// @brief  Build and append one record. Caller holds `wal_mutex_`.
    void append_record_(persistent_op_t op, vector_key_t key, char const* vector_or_null) {
        std::uint32_t payload_len = static_cast<std::uint32_t>( //
            1 + sizeof(vector_key_t) + (vector_or_null ? vector_bytes_ : 0));
        record_buffer_.resize(payload_len);
        record_buffer_[0] = static_cast<char>(op);
        std::memcpy(&record_buffer_[1], &key, sizeof(vector_key_t));
        if (vector_or_null)
            std::memcpy(&record_buffer_[1 + sizeof(vector_key_t)], vector_or_null, vector_bytes_);
        std::uint32_t crc = crc32_ieee(record_buffer_.data(), payload_len);
        std::FILE* out = wal_file_.get();
        std::fwrite(&payload_len, sizeof(payload_len), 1, out);
        std::fwrite(&crc, sizeof(crc), 1, out);
        std::fwrite(record_buffer_.data(), 1, payload_len, out);
        wal_bytes_ += sizeof(payload_len) + sizeof(crc) + payload_len;
    }

    /// @brief  Either recover from an existing manifest or initialize fresh.
    error_t open_(metric_t metric, index_dense_config_t index_config) {
        // Probe for an existing manifest. Absence = fresh open.
        std::uint64_t generation = 0;
        bool manifest_exists = false;
        if (file_t file{std::fopen(manifest_path_().c_str(), "rb")}) {
            persistent_manifest_t manifest{};
            if (std::fread(&manifest, 1, sizeof(manifest), file.get()) != sizeof(manifest))
                return error_t("Manifest is truncated");
            if (std::memcmp(manifest.magic, persistent_manifest_magic_k, 4) != 0)
                return error_t("Manifest magic mismatch");
            generation = manifest.generation;
            manifest_exists = true;
        }

        if (!manifest_exists) {
            // Fresh: empty index, snapshot.0, wal.0 (just header), manifest=0.
            typename index_t::state_result_t made = index_t::make(std::move(metric), index_config);
            if (!made)
                return std::move(made.error);
            index_ = std::move(made.index);
            if (!index_.try_reserve(config_.initial_capacity))
                return error_t("Failed to reserve initial capacity");
            refresh_metadata_();
            generation_ = 0;

            auto saved = index_.save(gen_path_(".snapshot", 0).c_str());
            if (!saved)
                return std::move(saved.error);

            std::string wal_path = gen_path_(".wal", 0);
            {
                file_t file{std::fopen(wal_path.c_str(), "wb")};
                if (!file)
                    return error_t("Failed to create WAL file");
                persistent_wal_header_t header = make_wal_header_();
                std::fwrite(&header, sizeof(header), 1, file.get());
                std::fflush(file.get());
            }

            if (error_t err = open_wal_for_append_(wal_path))
                return err;
            return write_manifest_(0);
        }

        // Recovery: load snapshot, reserve generously, replay WAL.
        generation_ = generation;
        std::string snapshot_path = gen_path_(".snapshot", generation_);
        typename index_t::state_result_t loaded = index_t::make(snapshot_path.c_str());
        if (!loaded)
            return std::move(loaded.error);
        index_ = std::move(loaded.index);
        refresh_metadata_();

        std::string wal_path = gen_path_(".wal", generation_);
        std::uint64_t wal_size = 0;
        if (file_t probe{std::fopen(wal_path.c_str(), "rb")}) {
            std::fseek(probe.get(), 0, SEEK_END);
            long s = std::ftell(probe.get());
            if (s > 0)
                wal_size = static_cast<std::uint64_t>(s);
        }
        // Upper bound on records that could fit in `wal_size`, used to size
        // capacity before replay (over-reserve is harmless).
        std::size_t wal_records_estimate = 0;
        if (wal_size > sizeof(persistent_wal_header_t))
            wal_records_estimate = (wal_size - sizeof(persistent_wal_header_t)) / (8 + 1 + sizeof(vector_key_t));
        if (!index_.try_reserve(index_.size() + wal_records_estimate + config_.initial_capacity))
            return error_t("Failed to reserve capacity for replay");

        // Replay every well-framed WAL record onto `index_`, stopping at the
        // first incomplete or crc-mismatched one. The header is validated
        // against the loaded snapshot's metadata.
        file_t wal{std::fopen(wal_path.c_str(), "rb")};
        if (!wal)
            return error_t("Failed to open WAL for replay");
        persistent_wal_header_t header{};
        if (std::fread(&header, 1, sizeof(header), wal.get()) != sizeof(header))
            return error_t("WAL header truncated");
        if (std::memcmp(header.magic, persistent_wal_magic_k, 4) != 0)
            return error_t("WAL magic mismatch");
        if (header.dimensions != dimensions_ || header.scalar_kind != scalar_kind_ ||
            header.metric_kind != metric_kind_)
            return error_t("WAL metadata does not match the snapshot");

        std::uint32_t const max_payload = static_cast<std::uint32_t>(1 + sizeof(vector_key_t) + vector_bytes_);
        std::vector<char> buffer;
        buffer.reserve(max_payload);
        while (true) {
            std::uint32_t len = 0, crc = 0;
            if (std::fread(&len, 1, sizeof(len), wal.get()) != sizeof(len))
                break;
            if (std::fread(&crc, 1, sizeof(crc), wal.get()) != sizeof(crc))
                break;
            if (len < 1 + sizeof(vector_key_t) || len > max_payload)
                break;
            buffer.resize(len);
            if (std::fread(buffer.data(), 1, len, wal.get()) != len)
                break;
            if (crc32_ieee(buffer.data(), len) != crc)
                break;

            // Replay is single-threaded: `contains` keeps add idempotent.
            persistent_op_t op = static_cast<persistent_op_t>(static_cast<std::uint8_t>(buffer[0]));
            vector_key_t key{};
            std::memcpy(&key, &buffer[1], sizeof(vector_key_t));
            if (op == persistent_op_add_k) {
                if (!index_.contains(key)) {
                    scalar_t const* vector = reinterpret_cast<scalar_t const*>(&buffer[1 + sizeof(vector_key_t)]);
                    auto r = index_.add(key, vector);
                    if (!r)
                        return std::move(r.error);
                }
            } else if (op == persistent_op_remove_k)
                index_.remove(key); // tolerant of an absent key
            else
                break; // unknown op
        }
        return open_wal_for_append_(wal_path);
    }

    /// @brief  Auto-trigger + step-driver, called from mutating paths.
    void maybe_drive_checkpoint_() {
        if (!checkpoint_active_.load(std::memory_order_acquire)) {
            if (ops_since_checkpoint_.load(std::memory_order_relaxed) >= config_.checkpoint_after_ops)
                start_checkpoint_();
        }
        if (checkpoint_active_.load(std::memory_order_acquire))
            drive_checkpoint_step_();
    }

    /// @brief  Begin barrier: drain in-flight applies under `wal_mutex_`, so
    ///         the snapshot covers exactly the WAL prefix up to `wal_bytes_`,
    ///         then `rebuild.begin`.
    void start_checkpoint_() {
        std::unique_lock<std::mutex> lock(wal_mutex_);
        if (checkpoint_active_.load(std::memory_order_acquire))
            return;

        // Holding `wal_mutex_` keeps new appends out; in-flight only drops.
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

    /// @brief  Drives one rebuild step; on the step that finishes the rebuild,
    ///         splices the WAL suffix onto a fresh `wal.<g+1>` and atomically
    ///         commits the manifest. Held briefly under `wal_mutex_` to keep
    ///         the suffix size stable.
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
        if (!rebuild_.finished())
            return;

        // Finish the checkpoint: copy the WAL suffix onto a fresh generation,
        // commit the manifest atomically, retire the old files.
        std::unique_lock<std::mutex> wal_lock(wal_mutex_);
        if (wal_file_)
            std::fflush(wal_file_.get());

        std::string old_wal = wal_path_;
        std::uint64_t suffix_start = checkpoint_watermark_;
        std::uint64_t suffix_end = wal_bytes_;
        std::string new_wal = gen_path_(".wal", checkpoint_gen_);

        {
            file_t dst{std::fopen(new_wal.c_str(), "wb")};
            if (!dst) {
                checkpoint_error_ = error_t("Failed to create new-gen WAL");
                return;
            }
            persistent_wal_header_t header = make_wal_header_();
            std::fwrite(&header, sizeof(header), 1, dst.get());

            if (suffix_end > suffix_start) {
                file_t src{std::fopen(old_wal.c_str(), "rb")};
                if (!src) {
                    std::remove(new_wal.c_str());
                    checkpoint_error_ = error_t("Failed to open old WAL for suffix copy");
                    return;
                }
                std::fseek(src.get(), static_cast<long>(suffix_start), SEEK_SET);
                char buffer[64 * 1024];
                std::uint64_t remaining = suffix_end - suffix_start;
                while (remaining > 0) {
                    std::size_t want =
                        remaining < sizeof(buffer) ? static_cast<std::size_t>(remaining) : sizeof(buffer);
                    std::size_t got = std::fread(buffer, 1, want, src.get());
                    if (got == 0)
                        break;
                    std::fwrite(buffer, 1, got, dst.get());
                    remaining -= got;
                }
            }
            std::fflush(dst.get());
        }

        // Swap the active WAL handle BEFORE the manifest commit: a crash here
        // still leaves the manifest pointing at the old generation and the
        // old WAL intact for recovery.
        wal_file_ = file_t{std::fopen(new_wal.c_str(), "ab")};
        if (!wal_file_) {
            checkpoint_error_ = error_t("Failed to reopen new WAL for append");
            return;
        }
        wal_path_ = new_wal;
        wal_bytes_ = sizeof(persistent_wal_header_t) + (suffix_end - suffix_start);

        // Atomic commit: from here the new generation is canonical.
        if (error_t err = write_manifest_(checkpoint_gen_)) {
            checkpoint_error_ = std::move(err);
            return;
        }

        // Retire the old generation. A crash before these removes leaves
        // orphans but recovery is correct - manifest already moved on.
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
