/**
 *  @file include/usearch/sequence/ram_index.hpp
 *  @author Ash Vardanian
 *  @date August 28, 2026
 *  @brief Similarity Search over variable-length byte sequences - text, DNA, or protein.
 *
 *  A stored payload is a @c sz_string_view_t into a @ref sequence_tape the caller reserves
 *  once, so a node costs sixteen bytes and no insertion ever allocates. Scoring is batched:
 *  one @c reset → @c push → @c drain round becomes a single sweep over the whole graph
 *  fan-out, through the metrics of @c ram_metric.hpp.
 *
 *  @section sequence_ram_index_concurrency Concurrency & Transactions
 *
 *  The tape is appended by any number of threads and never moves, so a view handed out at
 *  insertion stays valid for the tape's life. Each metric instance belongs to one thread;
 *  the index spreads rounds across threads.
 *
 *  @section sequence_ram_index_persistence Persistence
 *
 *  Checkpoints are SafeGraphs monoliths: keys, @c utf8 values, and neighbor lists, with the
 *  kernel the graph was built under recorded verbatim. Loading re-establishes ownership by
 *  appending every value into the codec's tape.
 *
 *  @section sequence_ram_index_organization File Organization
 *
 *  @b Storage - the tape. @b Serialization - the codec and the checkpoint verbs. @b Index -
 *  the alias over the basic graph index.
 */
#pragma once

#include <cstddef> // `std::size_t`
#include <cstdint> // `std::int64_t`

#include <atomic>      // `std::atomic` - the tape's append cursor
#include <memory>      // `std::allocator`
#include <string_view> // `std::string_view` - checkpoint paths & metric tokens
#include <utility>     // `std::move`, `std::forward`

#include "usearch/utilities/types.hpp"
#include "usearch/utilities/blocking_files.hpp" // `mapped_file_t` - a checkpoint's bytes
#include "usearch/utilities/ram_buffers.hpp"
#include "usearch/utilities/safegraphs_format.hpp" // SafeGraphs monoliths - the checkpoints
#include "usearch/basic/ram_index.hpp"
#include "usearch/sequence/ram_metric.hpp"

namespace unum {
namespace usearch {

#pragma region Storage

/**
 *  @brief The corpus: one byte buffer reserved up front, appended to by any number of
 *      threads, handing back views that stay valid for the tape's life.
 *
 *  Reserving exactly once is the whole trick - the buffer never moves, so a pointer handed
 *  out at insertion time never dangles, and a payload needs no allocator, no offset
 *  indirection, and no reference back to its owner. A @ref reserve after the first
 *  @ref append is refused rather than honored, and an @ref append past the reservation
 *  answers @c overflow_k rather than growing.
 */
template <typename allocator_type_ = std::allocator<byte_t>>
class sequence_tape {
  public:
    using allocator_t = allocator_type_;

    /** What one @ref append answers: where the bytes landed, or why they did not. */
    struct append_result_t {

        /** Success, or @c overflow_k once the reservation is exhausted. */
        status_t status = status_t::success_k;

        /** The stable view over the appended copy; empty on failure. */
        sz_string_view_t view {};

        explicit operator bool() const noexcept { return status == status_t::success_k; }
    };

  private:
    growing_buffer<char, allocator_t> bytes_;
    // Every append bumps it, so it takes its own line, away from the buffer every append and reader loads.
    alignas(default_alignment_k) std::size_t taken_ = 0;

  public:
    sequence_tape(allocator_t const& allocator = {}) noexcept : bytes_(allocator) {}
    sequence_tape(sequence_tape const&) = delete;
    sequence_tape& operator=(sequence_tape const&) = delete;

    /** @note Moving keeps every handed-out view valid - the buffer's address travels with it. */
    sequence_tape(sequence_tape&& other) noexcept
        : bytes_(std::move(other.bytes_)), taken_(std::exchange(other.taken_, 0)) {}

    sequence_tape& operator=(sequence_tape&& other) noexcept {
        bytes_ = std::move(other.bytes_);
        taken_ = std::exchange(other.taken_, 0);
        return *this;
    }

    /** Sizes the tape's single allocation; legal only before the first @ref append. */
    status_t reserve(std::size_t bytes) noexcept {
        if (standard_atomic_ref<std::size_t>(taken_).load(std::memory_order_acquire))
            return status_t::config_mismatch_k;
        return bytes_.resize_uninitialized(bytes);
    }

    /** Copies @p length bytes from @p data into the tape, answering a stable view. */
    append_result_t append(char const* data, std::size_t length) noexcept {
        std::size_t const offset = standard_atomic_ref<std::size_t>(taken_).fetch_add(length,
                                                                                      std::memory_order_acq_rel);
        if (offset + length > bytes_.size()) {
            standard_atomic_ref<std::size_t>(taken_).fetch_sub(length, std::memory_order_release);
            return {status_t::overflow_k, {}};
        }
        char* const destination = bytes_.data() + offset;
        for (std::size_t position = 0; position != length; ++position) destination[position] = data[position];
        return {status_t::success_k, sz_string_view_t {destination, length}};
    }

    /** Copies the bytes @p view spans into the tape, answering a stable view. */
    append_result_t append(sz_string_view_t view) noexcept { return append(view.start, view.length); }

    /** Bytes handed out so far. */
    std::size_t used_bytes() const noexcept {
        return standard_atomic_ref<std::size_t const>(taken_).load(std::memory_order_acquire);
    }

    /** Bytes the single reservation covers. */
    std::size_t reserved_bytes() const noexcept { return bytes_.size(); }
};

using sequence_tape_t = sequence_tape<>;

#pragma endregion Storage

#pragma region Serialization

/**
 *  @brief Saves a payload as its own bytes and restores it by appending into a tape.
 *
 *  Loading is where ownership is re-established: a stored view points into memory the
 *  index never owned, so the serialized bytes are copied into @p tape and the restored
 *  value views that copy.
 */
template <typename allocator_type_ = std::allocator<byte_t>>
class sequence_codec {
  public:
    using allocator_t = allocator_type_;
    using tape_t = sequence_tape<allocator_t>;

  private:
    tape_t* tape_ = nullptr;

  public:

    /** The interchange token of the elements a value emits: raw UTF-8 bytes. */
    inline static constexpr dtype_t value_dtype_k = dtype_t::utf8_k;

    explicit sequence_codec(tape_t& tape) noexcept : tape_(&tape) {}

    /** The tape restored values are appended into. */
    tape_t& tape() const noexcept { return *tape_; }

    std::size_t storage_requirements(sz_string_view_t const& value) const noexcept { return value.length; }

    status_t save(sz_string_view_t const& value, bytes_span_t output) const noexcept {
        if (output.size() < value.length) return status_t::overflow_k;
        for (std::size_t position = 0; position != value.length; ++position)
            output[position] = static_cast<std::byte>(value.start[position]);
        return status_t::success_k;
    }

    status_t load(bytes_view_t input, sz_string_view_t& value) const noexcept {
        typename tape_t::append_result_t const appended = tape_->append(reinterpret_cast<char const*>(input.data()),
                                                                        input.size());
        if (!appended) return appended.status;
        value = appended.view;
        return status_t::success_k;
    }
};

using sequence_codec_t = sequence_codec<>;

/**
 *  @brief Frames a sequence index as one SafeGraphs monolith over @p monolith - keys, @c utf8
 *      values, and neighbor lists - with @p metric recorded verbatim; the caller names the
 *      kernel the graph was built under, such as "levenshtein".
 *
 *  @param[inout] monolith The bytes the checkpoint lands in, at least
 *      @ref safegraphs_storage_requirements wide - a buffer, or a file's mapping.
 *  @param[in] index The graph to frame.
 *  @param[in] codec Encodes each stored view into its slice and names the column's dtype.
 *  @param[inout] executor Runs one region per thread.
 *  @param[in] config The alignment and provenance the header records.
 *  @param[in] metric The token the graph was built for.
 *  @param[in] progress Fires per finished region; answering @c cancel_k stops the rest.
 */
template <typename index_type_, typename allocator_type_, typename executor_type_,
          typename progress_type_ = dummy_progress_t>
status_t sequence_save(bytes_span_t monolith, index_type_ const& index, sequence_codec<allocator_type_> const& codec,
                       executor_type_& executor, save_config_t const& config, std::string_view metric,
                       progress_type_&& progress = {}) noexcept {
    return safegraphs_write_index(monolith, index, codec, executor, config, metric,
                                  std::forward<progress_type_>(progress));
}

/** Writes the monolith at @p path: sized from the index, created fresh, mapped, and
 *  flushed once every region landed. */
template <typename index_type_, typename allocator_type_, typename executor_type_,
          typename progress_type_ = dummy_progress_t>
status_t sequence_save(std::string_view path, index_type_ const& index, sequence_codec<allocator_type_> const& codec,
                       executor_type_& executor, save_config_t const& config, std::string_view metric,
                       progress_type_&& progress = {}) noexcept {
    auto const [bytes, measured] = safegraphs_storage_requirements(index, codec, config, metric);
    if (failed(measured)) return measured;
    auto [mapping, mapped] = mapped_file_t::write(path, bytes);
    if (failed(mapped)) return mapped;
    if (status_t const saved = sequence_save(mapping.span(), index, codec, executor, config, metric,
                                             std::forward<progress_type_>(progress));
        failed(saved))
        return saved;
    return mapping.sync();
}

/**
 *  @brief Restores a sequence index from the monolith @p monolith holds, reserving the codec's
 *      tape from the checkpoint's value bytes before any region appends: an untouched tape is
 *      sized to fit, a used one with too little room answers @c overflow_k.
 *
 *  @param[in] monolith The checkpoint's bytes - a buffer, or a file's mapping.
 *  @param[inout] index The graph to repopulate.
 *  @param[in] codec Decodes each slice into the tape it owns.
 *  @param[inout] executor Runs one region per thread.
 *  @param[in] config The reservation to take when the index has none.
 *  @param[in] progress Fires per finished region; answering @c cancel_k stops the rest.
 */
template <typename index_type_, typename allocator_type_, typename executor_type_,
          typename progress_type_ = dummy_progress_t>
status_t sequence_load(bytes_view_t monolith, index_type_& index, sequence_codec<allocator_type_> const& codec,
                       executor_type_& executor, load_config_t const& config, progress_type_&& progress = {}) noexcept {
    auto const [metadata, described] = safegraphs_read_index_metadata(monolith);
    if (failed(described)) return described;
    if (metadata.value_dtype != dtype_t::utf8_k) return status_t::config_mismatch_k;
    typename sequence_codec<allocator_type_>::tape_t& tape = codec.tape();
    std::size_t const needed = metadata.value_bytes;
    if (tape.used_bytes() != 0 && tape.reserved_bytes() - tape.used_bytes() < needed) return status_t::overflow_k;
    if (tape.used_bytes() == 0 && tape.reserved_bytes() < needed)
        if (status_t const reserved = tape.reserve(needed); failed(reserved)) return reserved;
    return safegraphs_read_index(monolith, index, codec, executor, config, std::forward<progress_type_>(progress));
}

/** Restores the index from the monolith at @p path, mapped read-only. */
template <typename index_type_, typename allocator_type_, typename executor_type_,
          typename progress_type_ = dummy_progress_t>
status_t sequence_load(std::string_view path, index_type_& index, sequence_codec<allocator_type_> const& codec,
                       executor_type_& executor, load_config_t const& config, progress_type_&& progress = {}) noexcept {
    auto [mapping, mapped] = mapped_file_t::read(path);
    if (failed(mapped)) return mapped;
    return sequence_load(mapping.view(), index, codec, executor, config, std::forward<progress_type_>(progress));
}

/** What the monolith @p monolith holds says about itself, the tape's byte need included. */
inline expected<safegraphs_index_metadata_t> sequence_metadata(bytes_view_t monolith) noexcept {
    return safegraphs_read_index_metadata(monolith);
}

/** The metadata of the monolith at @p path, mapped read-only. */
inline expected<safegraphs_index_metadata_t> sequence_metadata(std::string_view path) noexcept {
    auto [mapping, mapped] = mapped_file_t::read(path);
    if (failed(mapped)) return {safegraphs_index_metadata_t {}, mapped};
    return sequence_metadata(mapping.view());
}

#pragma endregion Serialization

#pragma region Index

/** Similarity Search over byte sequences: tape-backed views, integer distances, and a metric
 *  chosen per call - so a graph built under one kernel is queried under another without
 *  rebuilding. */
template <typename key_type_ = uuid_t, typename node_id_type_ = u32_t, typename wait_type_ = dummy_waiter_t,
          template <typename> class atomic_ref_template_ = standard_atomic_ref,
          typename allocator_type_ = std::allocator<byte_t>>
using sequence_ram_index = basic_ram_index<sz_string_view_t, key_type_, node_id_type_, i64_t, wait_type_,
                                           atomic_ref_template_, allocator_type_>;

using sequence_ram_index_t = sequence_ram_index<>;

#pragma endregion Index

} // namespace usearch
} // namespace unum
