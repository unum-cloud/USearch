/**
 *  @file include/usearch/utilities/nvme_walks.hpp
 *  @author Ash Vardanian
 *  @date July 22, 2026
 *  @brief The storage walk every NVMe tier drives: the parts' vocabulary, and one thread's walks as
 *      a pure state machine of reads.
 *
 *  A walk never performs I/O itself: it only emits read requests and consumes completions, and the
 *  engines of @c nvme_device.hpp submit them. What a landed value row means belongs to the tier
 *  that drives the walk, so the walker names only the tier's own vocabulary:
 *
 *  - @c engine_t, @c arena_t and @c medium_t, the executor's engine, arena and medium;
 *  - @c node_id_t, @c search_result_t, @c scratch_allocator_t and @c results_array_t;
 *  - @c part_t, where one part keeps its columns, with @c nodes, @c connectivity as its widest
 *    neighbors row, and @c key_dtype;
 *  - @c row_offset_ and @c row_length_, which locate any row of a part's key, value or neighbors
 *    column;
 *  - @c walk_t, one walk's pools, and @c walk_shape_t, the call's sizes;
 *  - @c prime_scoring_, which prepares a row's query once the row primes;
 *  - @c score_staged_, which scores a walk's staged flights, folds and releases them, and ranks the
 *    rest through the walker's @c rank_.
 *
 *  Nothing here reads a value row, so the file needs no kernel library, and every tier compiles the
 *  same walk.
 */
#pragma once

#include <cassert> // `assert`
#include <cstring> // `std::memset`

#include <array>       // `std::array`
#include <atomic>      // `std::atomic_ref`
#include <bit>         // `std::bit_cast`
#include <chrono>      // `std::chrono::steady_clock`, for per-row latencies
#include <new>         // placement `new`
#include <span>        // `std::span`
#include <string_view> // `std::string_view`

#include "usearch/utilities/types.hpp"
#include "usearch/utilities/nvme_device.hpp"       // `io_request_t`, `io_completion_t`, `block_bytes_of`
#include "usearch/utilities/ram_buffers.hpp"       // `growing_buffer`
#include "usearch/utilities/ram_device.hpp"        // `forkunion_allocator_t`
#include "usearch/utilities/ram_tables.hpp"        // `membership_t`
#include "usearch/utilities/safegraphs_format.hpp" // `safegraphs_read_metadata`, `safegraphs_widened_key`

namespace unum::usearch::utilities {

#pragma region Parts

/** Where one SafeGraphs part keeps its columns, read once from its header: what a walk needs to
 *  address any row, and nothing per node. */
struct nvme_part_t {

    /** Rows the part holds; neighbor IDs at or past it are sentinels. */
    u32_t nodes = 0;

    /** Logical dimensions of one value row. */
    u32_t dimensions = 0;

    /** Neighbor IDs one neighbors row holds. */
    u32_t connectivity = 0;

    /** Bytes of one key row. */
    u32_t key_row_bytes = 0;

    /** The key column's dtype, which @ref safegraphs_widened_key reads each row as. */
    dtype_t key_dtype = dtype_t::unknown_k;

    /** Bytes of one value row. */
    u32_t value_row_bytes = 0;

    /** Where row zero of the key column starts, in bytes from the part's own start. */
    u64_t keys_begin = 0;

    /** Where row zero of the value column starts. */
    u64_t values_begin = 0;

    /** Where row zero of the neighbors column starts. */
    u64_t neighbors_begin = 0;
};

/** Where one entry lives among the shards a call reads: what a walk ranks by until keys resolve. */
template <typename node_id_type_>
struct nvme_locator {

    /** The shard holding the entry, positional among the call's shards. */
    u32_t shard = 0;

    /** The entry's ID inside that shard's own space. */
    node_id_type_ node = 0;
};

/** Maps a walked node ID of one shard to its locator, as an ID resolver and as an ID-to-key
 *  table. */
template <typename node_id_type_>
struct nvme_shard_locators {

    /** The shard every resolved ID belongs to. */
    u32_t shard = 0;

    nvme_locator<node_id_type_> operator()(node_id_type_ node) const noexcept { return {shard, node}; }
    nvme_locator<node_id_type_> operator[](node_id_type_ node) const noexcept { return {shard, node}; }
};

/** Reads of one purpose a call issued, and the bytes they transferred, block rounding included. */
struct nvme_reads_t {

    /** Reads issued. */
    std::size_t count = 0;

    /** Bytes those reads transferred. */
    std::size_t bytes = 0;

    nvme_reads_t& operator+=(nvme_reads_t const& other) noexcept {
        count += other.count, bytes += other.bytes;
        return *this;
    }
};

/** The bytes the widest block-aligned read covering any @p row_bytes -byte row takes on
 *  @p block_bytes blocks. */
inline u64_t nvme_covering_bytes(u64_t row_bytes, u64_t block_bytes) noexcept {
    return round_up_to_multiple<u64_t>(row_bytes + block_bytes - 1, block_bytes);
}

/** The read of whole @p block_bytes blocks covering the @p bytes at @p begin on medium @p medium,
 *  landing in @p buffer, its completion carrying @p context back. */
inline io_request_t nvme_covering_read(u64_t begin, u64_t bytes, u64_t block_bytes, io_medium_id_t medium,
                                       io_buffer_id_t buffer, u64_t context) noexcept {
    u64_t const first = begin - begin % block_bytes;
    u64_t const bound = round_up_to_multiple<u64_t>(begin + bytes, block_bytes);
    assert(bound - first <= io_max_length_k && "A covering read must fit the length a completion counts");
    return {first, static_cast<u32_t>(bound - first), medium, buffer, context};
}

/** Reads the prefix and the header of the part @p source holds, any reader with
 *  `read_at(offset, bytes)`, into @p header_bytes; @c malformed_k when it holds no monolith. */
template <typename source_type_, typename allocator_type_>
    requires requires(source_type_ const& source, bytes_span_t bytes) { source.read_at(0, bytes); }
status_t nvme_read_header(source_type_ const& source, growing_buffer<byte_t, allocator_type_>& header_bytes) noexcept {
    std::array<byte_t, sizeof(safegraphs_prefix_t)> prefix_bytes {};
    if (status_t const read = source.read_at(0, prefix_bytes); failed(read)) return status_t::malformed_k;
    safegraphs_prefix_t const prefix = std::bit_cast<safegraphs_prefix_t>(prefix_bytes);
    if (prefix.magic != safegraphs_prefix_t::magic_k) return status_t::malformed_k;
    std::size_t const header_length = static_cast<std::size_t>(little_endian(prefix.header_length));
    if (failed(header_bytes.resize_uninitialized(sizeof(prefix) + header_length))) return status_t::bad_alloc_k;
    if (status_t const read = source.read_at(0, to_writable_span(header_bytes)); failed(read))
        return status_t::malformed_k;
    return status_t::success_k;
}

/** Reads the prefix and the header of the part at @p path into @p header_bytes through the medium
 *  a view reads it by: a file or a device, or in SPDK builds a namespace a transport ID names,
 *  each with an optional @c base: token; @c malformed_k when no monolith starts there. */
template <typename allocator_type_>
status_t nvme_read_header(std::string_view path, growing_buffer<byte_t, allocator_type_>& header_bytes) noexcept {
#if USEARCH_WITH_LINUX_SPDK
    if (names_transport_id(path)) {
        auto [name_space, open_status] = spdk_medium_t::open(path);
        if (failed(open_status)) return open_status;
        return nvme_read_header(name_space, header_bytes);
    }
#endif
    if (names_transport_id(path)) return status_t::unsupported_k;
    auto [file, open_status] = file_medium_t::open(path);
    if (failed(open_status)) return open_status;
    return nvme_read_header(file, header_bytes);
}

/** The metadata of the checkpoint at @p path without viewing it: a manifest in a file, or a
 *  monolith wherever @ref nvme_read_header reaches one, its header read as a view reads it. */
inline expected<safegraphs_metadata_t> nvme_read_metadata(std::string_view path) noexcept {
    growing_buffer<byte_t> header_bytes {global_allocator};
    status_t const read = nvme_read_header(path, header_bytes);
    if (read == status_t::malformed_k && !names_transport_id(path)) return safegraphs_read_manifest(path);
    if (failed(read)) return {safegraphs_metadata_t {}, read};
    return safegraphs_read_metadata(to_span(header_bytes));
}

#if USEARCH_OS_WINDOWS_

/** Whether @p path names a directory, which a relocation joins a part's file name onto. */
inline bool nvme_is_directory_(char const* path) noexcept {
    DWORD const attributes = ::GetFileAttributesA(path);
    return attributes != INVALID_FILE_ATTRIBUTES && (attributes & FILE_ATTRIBUTE_DIRECTORY);
}

#else

/** Whether @p path names a directory, which a relocation joins a part's file name onto. */
inline bool nvme_is_directory_(char const* path) noexcept {
    struct stat description;
    return ::stat(path, &description) == 0 && S_ISDIR(description.st_mode);
}

#endif

/** Writes @p directory and @p name joined by one separator into @p buffer, terminated for the
 *  OS. */
inline status_t nvme_join_path_(growing_buffer<char, forkunion_allocator_t>& buffer, std::string_view directory,
                                std::string_view name) noexcept {
    std::size_t const separators = directory.empty() || directory.ends_with('/') || directory.ends_with('\\') ? 0 : 1;
    std::size_t const length = directory.size() + separators + name.size();
    if (failed(buffer.resize_uninitialized(length + 1))) return status_t::bad_alloc_k;
    for (std::size_t position = 0; position != directory.size(); ++position) buffer[position] = directory[position];
    if (separators) buffer[directory.size()] = '/';
    for (std::size_t position = 0; position != name.size(); ++position)
        buffer[directory.size() + separators + position] = name[position];
    buffer[length] = '\0';
    return status_t::success_k;
}

/** Which keys a walk reads besides the value and neighbors rows. */
enum class nvme_key_reads_t : u8_t {

    /** Only the outputs a row writes resolve into keys, one read each, once its walks finished. */
    outputs_k,

    /** Every scored row's key lands beside its value, for a fold or a rescoring that reads keys. */
    scored_k,
};

#pragma endregion Parts

#pragma region Walks

/** Where a walk stands between its reads. */
enum class nvme_walk_stage_t : u8_t {

    /** Finished, its pool merged into its row, or never primed. */
    finished_k,

    /** Its landmarks are in flight in hop slot zero, and every row that lands enters the
     *  frontier; spare hop slots may already expand the landmarks that landed. */
    seeding_k,

    /** Its landmarks all landed, and its expansions, or the scores they issued, are in
     *  flight. */
    expanding_k,

    /** A sated fold or a refused re-table stopped it; its outstanding reads land unscored, then
     *  it finishes. */
    draining_k,
};

/** Why a walk's read was issued, which routes its landing. */
enum class nvme_purpose_t : u8_t {

    /** A value row, scored against the query. */
    score_k,

    /** A neighbors row, extending the frontier. */
    expand_k,

    /** A key row, for a keyed fold or a written result. */
    key_k,
};

/** One scored, expanded or resolved row in flight. Its first read lands in the buffer of the
 *  flight's own index, and a keyed score's key in the buffer one flight count past it. The
 *  shard it reads follows from its issuer, which keeps a flight of 32-bit IDs at 16 bytes. */
template <typename node_id_type_>
struct nvme_flight {

    /** The walk that issued a score or an expansion, or the row slot behind a resolution. */
    u32_t issuer = 0;

    /** The row's node ID inside its shard. */
    node_id_type_ node = 0;

    /** Where in its issuer the landing counts: a walk's hop slot, or a row slot's output. */
    u32_t position = 0;

    /** Reads still to land: two for a keyed score, one otherwise. */
    u8_t reads_left = 0;

    /** What its landing does: @c score_k scores, @c expand_k expands, @c key_k resolves. */
    nvme_purpose_t purpose = nvme_purpose_t::score_k;
};

static_assert(sizeof(nvme_flight<u32_t>) == 16, "A flight is 16 bytes, so none straddles a cache line");

/** Where a row slot stands. */
enum class nvme_row_stage_t : u8_t {

    /** Free for the next row. */
    idle_k,

    /** Some of its walks are still going. */
    walking_k,

    /** Every walk finished, and its written locators are resolving into keys. */
    resolving_k,
};

/** One row in flight on a thread: its walks still going, and its resolution's progress. */
struct nvme_row_slot_t {

    /** Walks of the row not finished yet. */
    u32_t walks_left = 0;

    /** Locators the row asks to resolve. */
    u32_t outputs = 0;

    /** The next locator to issue a key read for. */
    u32_t next_output = 0;

    /** Key reads of the row in flight. */
    u32_t resolving = 0;

    /** Where the slot stands. */
    nvme_row_stage_t stage = nvme_row_stage_t::idle_k;

    /** The batch row the slot holds. */
    std::size_t row = 0;

    /** When the row's walks primed, which its latency counts from; unset when the call collects
     *  none. */
    std::chrono::steady_clock::time_point primed;
};

/** One call's hold on an executor: the claim, then the media, which detach before the claim
 *  lifts. */
template <typename executor_type_>
struct nvme_call {

    /** The executor's one-caller hold. */
    typename executor_type_::claim_t claim;

    /** The media every engine reads until the call ends. */
    typename executor_type_::attachment_t attachment;
};

/** Claims @p executor, sizes the arena of every thread at @p scratch_bytes and every engine at
 *  @p buffers staging buffers of @p buffer_bytes, and attaches @p media. Nothing rewinds the
 *  arenas again before the call ends, so whatever a call carves lasts all of it. */
template <typename executor_type_>
expected<nvme_call<executor_type_>> nvme_begin_call(executor_type_& executor,
                                                    std::span<typename executor_type_::medium_t const> media,
                                                    std::size_t scratch_bytes, std::size_t buffers,
                                                    std::size_t buffer_bytes) noexcept {
    nvme_call<executor_type_> call;
    auto [claim, claim_status] = executor.claim();
    if (failed(claim_status)) return {std::move(call), claim_status};
    call.claim = std::move(claim);
    if (status_t const reserved = executor.reserve(scratch_bytes, buffers, buffer_bytes); failed(reserved))
        return {std::move(call), reserved};
    auto [attachment, attach_status] = executor.attach(media);
    call.attachment = std::move(attachment);
    return {std::move(call), attach_status};
}

/**
 *  @brief One thread's share of a call: its walks, its rows, its flights and the reads not yet
 *      enqueued, driven until the thread's rows run out.
 *
 *  @tparam tier_type_ The index whose vocabulary the walk speaks and whose @c score_staged_ scores
 *      its landed value rows; it befriends the walker, as the walker befriends it.
 *  @tparam row_type_ One row of the batch: its query, its fold, its bound and its outputs.
 */
template <typename tier_type_, typename row_type_>
class row_walker {
    friend tier_type_;

    using tier_t = tier_type_;
    using engine_t = typename tier_t::engine_t;
    using arena_t = typename tier_t::arena_t;
    using medium_t = typename tier_t::medium_t;
    using part_t = typename tier_t::part_t;
    using node_id_t = typename tier_t::node_id_t;
    using flight_t = nvme_flight<node_id_t>;
    using walk_t = typename tier_t::walk_t;
    using walk_shape_t = typename tier_t::walk_shape_t;
    using search_result_t = typename tier_t::search_result_t;
    using scratch_allocator_t = typename tier_t::scratch_allocator_t;
    using results_array_t = typename tier_t::results_array_t;
    using scored_node_t = typename results_array_t::value_type;

  public:
    using row_t = row_type_;
    using key_t = typename row_t::key_t;

    row_walker(engine_t& engine, arena_t& arena, std::span<medium_t const> media, std::span<part_t const> parts,
               walk_shape_t const& shape) noexcept
        : engine_(engine), arena_(arena), media_(media), parts_(parts), shape_(shape) {}

    row_walker(row_walker const&) = delete;
    row_walker& operator=(row_walker const&) = delete;

    ~row_walker() noexcept {
        for (walk_t& walk : walks_) walk.~walk_t();
        for (row_t& row : rows_) row.~row_t();
    }

    /** Carves every walk, row and flight of the thread, each row built from @p inputs. */
    status_t reserve(typename row_t::inputs_t const& inputs) noexcept {
        std::size_t const parts = parts_.size(), rows = shape_.rows_per_thread;
        std::size_t const walks = rows * parts, flights = walks * shape_.walk_flights;
        walk_t* const walk_slots = carve_<walk_t>(walks);
        row_t* const row_slots = carve_<row_t>(rows);
        nvme_row_slot_t* const slot_states = carve_<nvme_row_slot_t>(rows);
        flight_t* const flight_states = carve_<flight_t>(flights);
        u32_t* const free_flights = carve_<u32_t>(flights);
        u32_t* const hop_flights = carve_<u32_t>(walks * shape_.hops_per_walk);
        io_request_t* const backlog = carve_<io_request_t>(shape_.buffers);
        io_completion_t* const completions = carve_<io_completion_t>(shape_.buffers);
        if (!walk_slots || !row_slots || !slot_states || !flight_states || !free_flights || !hop_flights || !backlog ||
            !completions)
            return status_t::bad_alloc_k;

        scratch_allocator_t const allocator {arena_};
        std::size_t const expansion_factor = shape_.expansion_factor, connectivity = parts_[0].connectivity;
        for (std::size_t walk = 0; walk != walks; ++walk) {
            walk_t& constructed = *new (walk_slots + walk) walk_t {allocator};
            walks_ = {walk_slots, walk + 1};
            if (failed(constructed.results.reserve(expansion_factor)) ||
                failed(constructed.frontier.reserve(expansion_factor * 4)) ||
                failed(constructed.visited.reserve(expansion_factor * connectivity)))
                return status_t::bad_alloc_k;
        }
        for (std::size_t row = 0; row != rows; ++row) {
            row_t& constructed = *new (row_slots + row) row_t {inputs, allocator};
            rows_ = {row_slots, row + 1};
            if (status_t const reserved = constructed.reserve(parts, connectivity); failed(reserved)) return reserved;
        }
        for (std::size_t row = 0; row != rows; ++row) new (slot_states + row) nvme_row_slot_t {};
        for (std::size_t flight = 0; flight != flights; ++flight)
            free_flights[flight] = static_cast<u32_t>(flights - 1 - flight);
        slots_ = {slot_states, rows};
        flights_ = {flight_states, flights};
        free_flights_ = {free_flights, flights};
        hop_flights_ = {hop_flights, walks * shape_.hops_per_walk};
        free_count_ = flights;
        backlog_ = {backlog, shape_.buffers};
        completions_ = {completions, shape_.buffers};
        return status_t::success_k;
    }

    /** Primes rows until the batch runs out, keeping every slot's walks in flight, and drains the
     *  engine before it returns, whatever happened; a failure on any thread stops this one too.
     *  Answers the first row that refused, since a failure is published to every thread
     *  instead. */
    status_t run() noexcept {
        while (true) {
            if (succeeded(failure_)) {
                if (status_t const shared = shared_failure_(); failed(shared)) failure_ = shared;
                else prime_rows_();
            }
            if (failed(failure_) && !outstanding_) break;
            if (succeeded(failure_)) {
                if (!rows_in_flight_ && exhausted_()) break;
                enqueue_backlog_();
            }
            if (!outstanding_) {
                assert((failed(failure_) || !rows_in_flight_) && "A row in flight always waits on a read");
                continue;
            }
            if (status_t const submitted = engine_.submit(io_wait_t::blocking_k); failed(submitted)) {
                // A refused submission returns with nothing in flight and nothing left to poll.
                fail(submitted);
                outstanding_ = 0;
                break;
            }
            telemetry_.round_trips += rows_in_flight_;
            for (std::size_t polled = engine_.poll_completions(completions_); polled;
                 polled = engine_.poll_completions(completions_)) {
                // Completions of walk reads move to the front, as their flights may serve
                // other rows by the end of the batch.
                std::size_t advancing = 0;
                for (io_completion_t const& completion : completions_.first(polled))
                    if (land_(completion)) completions_[advancing++] = completion;
                // Walks advance once the whole batch landed, so their next hops weigh every row
                // that did.
                if (succeeded(failure_))
                    for (io_completion_t const& completion : completions_.first(advancing))
                        advance_(static_cast<u32_t>(completion.context));
            }
        }
        return refusal_;
    }

    /** What the thread counted. */
    search_result_t const& telemetry() const noexcept { return telemetry_; }

    /** Records the thread's first failure and publishes it, so every thread stops priming
     *  rows. */
    void fail(status_t failure) noexcept {
        if (succeeded(failure_)) failure_ = failure;
        status_t unset = status_t::success_k;
        std::atomic_ref<status_t>(*shape_.failure)
            .compare_exchange_strong(unset, failure, std::memory_order_relaxed, std::memory_order_relaxed);
    }

    /**
     *  @brief Walks every row of a batch over every part in @p parts, the rows' walks interleaved
     *      on every thread of @p executor, each row built from @p inputs as @p row_type_ says.
     *
     *  Runs inside a call @ref nvme_begin_call began at @p shape 's sizes, and sums the threads'
     *  telemetry; @p counts are the rows' own, summed into the result's @c count. @p latencies
     *  takes one duration per row, or is empty.
     */
    template <typename executor_type_>
    static expected<search_result_t> run_walks(std::span<medium_t const> media, std::span<part_t const> parts,
                                               executor_type_& executor, walk_shape_t shape,
                                               typename row_t::inputs_t const& inputs,
                                               std::span<std::size_t const> counts,
                                               std::span<u64_t> latencies) noexcept {
        search_result_t merged;
        if (!latencies.empty() && latencies.size() != shape.rows_count) return {merged, status_t::config_mismatch_k};
        alignas(default_alignment_k) std::size_t cursor = 0;
        status_t failure = status_t::success_k;
        shape.cursor = &cursor;
        shape.failure = &failure;
        shape.latencies = latencies;
        status_t refusal = status_t::success_k;
        auto add = [](std::size_t& into, std::size_t value) noexcept {
            std::atomic_ref<std::size_t>(into).fetch_add(value, std::memory_order_relaxed);
        };
        auto add_reads = [&](nvme_reads_t& into, nvme_reads_t const& reads) noexcept {
            add(into.count, reads.count), add(into.bytes, reads.bytes);
        };
        auto walk_thread = [&](std::size_t thread, std::size_t) noexcept {
            row_walker walker {executor.engine(thread), executor.arena(thread), media, parts, shape};
            status_t const reserved = walker.reserve(inputs);
            if (failed(reserved)) walker.fail(reserved);
            status_t const refused = walker.run();
            search_result_t const& telemetry = walker.telemetry();
            add(merged.visited_members, telemetry.visited_members);
            add(merged.computed_distances, telemetry.computed_distances);
            add(merged.restarts, telemetry.restarts);
            add(merged.saturated_walks, telemetry.saturated_walks);
            add_reads(merged.neighbor_reads, telemetry.neighbor_reads);
            add_reads(merged.value_reads, telemetry.value_reads);
            add_reads(merged.key_reads, telemetry.key_reads);
            add(merged.round_trips, telemetry.round_trips);
            status_t unset = status_t::success_k;
            if (failed(refused))
                std::atomic_ref<status_t>(refusal).compare_exchange_strong(unset, refused, std::memory_order_relaxed,
                                                                           std::memory_order_relaxed);
        };
        if (status_t const fanned = executor.for_threads(walk_thread); failed(fanned)) return {merged, fanned};
        for (std::size_t const count : counts) merged.count += count;
        return {merged, failed(failure) ? failure : refusal};
    }

  private:
    template <typename value_type_>
    value_type_* carve_(std::size_t count) noexcept {
        return reinterpret_cast<value_type_*>(
            arena_.allocate(max_of<std::size_t>(count, 1) * sizeof(value_type_), default_alignment_k));
    }

    status_t shared_failure_() const noexcept {
        return std::atomic_ref<status_t>(*shape_.failure).load(std::memory_order_relaxed);
    }

    bool exhausted_() const noexcept {
        return std::atomic_ref<std::size_t>(*shape_.cursor).load(std::memory_order_relaxed) >= shape_.rows_count;
    }

    u32_t take_flight_() noexcept { return free_flights_[--free_count_]; }
    void release_flight_(u32_t flight) noexcept { free_flights_[free_count_++] = flight; }

    /** The flights each hop slot of @p walk_index waits on; a zero slot is free. */
    std::span<u32_t> hop_flights_of_(u32_t walk_index) noexcept {
        return hop_flights_.subspan(walk_index * shape_.hops_per_walk, shape_.hops_per_walk);
    }

    /** Whether a hop slot of @p walk_index still waits on a flight. */
    bool in_flight_(u32_t walk_index) noexcept {
        for (u32_t const flights : hop_flights_of_(walk_index))
            if (flights) return true;
        return false;
    }

    /** Stops @p walk after a scratch refusal, counted as saturated: its outstanding reads land
     *  unscored, then it finishes. */
    void saturate_(walk_t& walk) noexcept {
        ++telemetry_.saturated_walks;
        walk.stage = nvme_walk_stage_t::draining_k;
    }

    /** The buffer a keyed score's key lands in: one flight count past its value's. */
    io_buffer_id_t key_buffer_(u32_t flight_index) const noexcept {
        return static_cast<io_buffer_id_t>(flight_index + flights_.size());
    }

    /** The shard @p flight reads: its walk's, or the shard of the output a resolution fills. */
    u32_t shard_of_(flight_t const& flight) const noexcept {
        return flight.purpose == nvme_purpose_t::key_k ? rows_[flight.issuer].locator(flight.position).shard
                                                       : walks_[flight.issuer].shard;
    }

    /** Where the row of @p flight starts in the column @p purpose reads, from the medium's byte
     *  zero. */
    u64_t row_begin_(flight_t const& flight, nvme_purpose_t purpose) const noexcept {
        u32_t const shard = shard_of_(flight);
        return byte_base_of(media_[shard]) + tier_t::row_offset_(parts_[shard], purpose, flight);
    }

    u64_t row_bytes_(flight_t const& flight, nvme_purpose_t purpose) const noexcept {
        return tier_t::row_length_(parts_[shard_of_(flight)], purpose, flight);
    }

    /** The landed row of @p flight_index in the column @p purpose read, inside @p buffer. */
    byte_t const* landed_row_(u32_t flight_index, nvme_purpose_t purpose, io_buffer_id_t buffer) noexcept {
        flight_t const& flight = flights_[flight_index];
        return engine_.buffer(buffer) + row_begin_(flight, purpose) % block_bytes_of(media_[shard_of_(flight)]);
    }

    /** Stages the covering read of @p flight_index 's row in the column @p purpose reads, into
     *  @p buffer, its context naming the flight's issuer. */
    void issue_(u32_t flight_index, nvme_purpose_t purpose, io_buffer_id_t buffer) noexcept {
        flight_t const& flight = flights_[flight_index];
        u32_t const shard = shard_of_(flight);
        io_request_t const request = nvme_covering_read(row_begin_(flight, purpose), row_bytes_(flight, purpose),
                                                        block_bytes_of(media_[shard]), shard, buffer, flight.issuer);
        backlog_[backlog_size_++] = request;
        nvme_reads_t& reads = purpose == nvme_purpose_t::expand_k ? telemetry_.neighbor_reads
                              : purpose == nvme_purpose_t::key_k  ? telemetry_.key_reads
                                                                  : telemetry_.value_reads;
        reads += nvme_reads_t {1, request.length};
    }

    /** Hands the staged reads to the engine until it holds no more; the rest wait for
     *  completions. */
    void enqueue_backlog_() noexcept {
        while (backlog_size_) {
            status_t const enqueued = engine_.enqueue_read(backlog_[backlog_size_ - 1]);
            if (enqueued == status_t::overflow_k && outstanding_) return;
            // A full engine holding none of this thread's reads would never drain, so the
            // thread fails instead.
            if (failed(enqueued)) return fail(enqueued == status_t::overflow_k ? status_t::io_k : enqueued);
            --backlog_size_;
            ++outstanding_;
        }
    }

    /** Takes the next rows of the batch into every idle slot and seeds all of their walks. */
    void prime_rows_() noexcept {
        std::size_t const parts = parts_.size();
        for (u32_t slot_index = 0; slot_index != slots_.size(); ++slot_index) {
            nvme_row_slot_t& slot = slots_[slot_index];
            if (slot.stage != nvme_row_stage_t::idle_k) continue;
            std::size_t const row =
                std::atomic_ref<std::size_t>(*shape_.cursor).fetch_add(1, std::memory_order_relaxed);
            if (row >= shape_.rows_count) return;
            slot = nvme_row_slot_t {static_cast<u32_t>(parts), 0, 0, 0, nvme_row_stage_t::walking_k, row, {}};
            if (!shape_.latencies.empty()) slot.primed = std::chrono::steady_clock::now();
            ++rows_in_flight_;
            row_t& primed = rows_[slot_index];
            primed.prime(row);
            tier_t::prime_scoring_(shape_, primed);
            for (u32_t shard = 0; shard != parts; ++shard) {
                u32_t const walk_index = slot_index * static_cast<u32_t>(parts) + shard;
                walk_t& walk = walks_[walk_index];
                walk.results.clear();
                walk.frontier.clear();
                walk.visited.clear();
                walk.staged.clear();
                walk.random_generator = splitmix64_t {mix_bits(u64_t(row) * parts + shard + 1)};
                walk.shard = shard;
                walk.row_slot = slot_index;
                walk.stage = nvme_walk_stage_t::seeding_k;
                std::span<u32_t> const hop_flights = hop_flights_of_(walk_index);
                std::memset(hop_flights.data(), 0, hop_flights.size_bytes());
                std::size_t const landmarks = round_up_to_multiple<std::size_t>(parts_[shard].connectivity, 4);
                if (!seed_(walk_index, landmarks)) finish_walk_(walk_index);
            }
        }
    }

    /** Issues up to @p count fresh landmarks of walk @p walk_index as one hop, answering how
     *  many. A walk seeds only with nothing in flight, so the hop takes slot zero. */
    std::size_t seed_(u32_t walk_index, std::size_t count) noexcept {
        walk_t& walk = walks_[walk_index];
        node_id_t const nodes = parts_[walk.shard].nodes;
        if (failed(walk.visited.reserve(walk.visited.size() + count))) return saturate_(walk), 0;
        std::size_t issued = 0;
        for (; issued != count && walk.visited.size() < nodes; ++issued) {
            node_id_t landmark = uniform_below<node_id_t> {nodes}(walk.random_generator);
            while (walk.visited.contains(landmark)) landmark = landmark + 1 == nodes ? 0 : landmark + 1;
            [[maybe_unused]] membership_t const marked = walk.visited.insert(landmark);
            issue_score_(walk_index, landmark, 0);
        }
        std::span<u32_t> const hop_flights = hop_flights_of_(walk_index);
        hop_flights[0] = static_cast<u32_t>(issued);
        return issued;
    }

    void issue_score_(u32_t walk_index, node_id_t node, u32_t hop) noexcept {
        u32_t const flight = take_flight_();
        flights_[flight] = flight_t {walk_index, node, hop, static_cast<u8_t>(shape_.buffers_per_flight),
                                     nvme_purpose_t::score_k};
        issue_(flight, nvme_purpose_t::score_k, flight);
        if constexpr (row_t::key_reads_k == nvme_key_reads_t::scored_k)
            issue_(flight, nvme_purpose_t::key_k, key_buffer_(flight));
    }

    /** Routes one completion to its flight, answering whether its issuer is a walk the batch's
     *  end advances; a read short of its row fails the thread. */
    bool land_(io_completion_t const& completion) noexcept {
        --outstanding_;
        if (failed(failure_)) return false;
        u32_t const flight_index = static_cast<u32_t>(completion.buffer % flights_.size());
        flight_t& flight = flights_[flight_index];
        nvme_purpose_t const purpose = completion.buffer == flight_index ? flight.purpose : nvme_purpose_t::key_k;
        u64_t const row_begin = row_begin_(flight, purpose);
        u64_t const needed = row_begin % block_bytes_of(media_[shard_of_(flight)]) + row_bytes_(flight, purpose);
        if (completion.result < 0 || static_cast<u64_t>(completion.result) < needed) return fail(status_t::io_k), false;
        if (--flight.reads_left) return false;
        switch (flight.purpose) {
        case nvme_purpose_t::score_k: {
            walk_t& walk = walks_[flight.issuer];
            std::span<u32_t> const hop_flights = hop_flights_of_(flight.issuer);
            [[maybe_unused]] fu::status_t const queued = walk.staged.push_back(flight_index);
            // A quad scores once full or once a hop's last row lands, whichever hops its lanes
            // came from.
            if (--hop_flights[flight.position] && !walk.staged.full()) return true;
            flush_(walk);
            // Landmarks take hop slot zero, so seeding ends once that slot drains.
            if (walk.stage == nvme_walk_stage_t::seeding_k && !hop_flights[0])
                walk.stage = nvme_walk_stage_t::expanding_k;
            return true;
        }
        case nvme_purpose_t::expand_k: expand_(flight.issuer, flight_index); return true;
        case nvme_purpose_t::key_k: {
            u32_t const slot_index = flight.issuer;
            byte_t const* const stored = landed_row_(flight_index, nvme_purpose_t::key_k, flight_index);
            dtype_t const stored_dtype = parts_[shard_of_(flight)].key_dtype;
            rows_[slot_index].resolve(flight.position, safegraphs_widened_key<key_t>(stored, stored_dtype));
            release_flight_(flight_index);
            --slots_[slot_index].resolving;
            resolve_row_(slot_index);
            return false;
        }
        }
        return false;
    }

    /** Releases the staged flights of a draining @p walk unscored, or hands them to the tier, which
     *  scores them as one quad, folds them through the walk's row, releases them and ranks the rest
     *  through @ref rank_. */
    void flush_(walk_t& walk) noexcept {
        if (!walk.staged.size()) return;
        if (walk.stage == nvme_walk_stage_t::draining_k) {
            for (u32_t const flight : walk.staged) release_flight_(flight);
            return walk.staged.clear();
        }
        tier_t::score_staged_(*this, walk);
    }

    /** Merges the ascending @p scored rows of @p walk into its pool, and queues them for expansion:
     *  every landmark, as the in-memory walk seeds the frontier, and a neighbor only when the pool
     *  took it. */
    [[gnu::always_inline]] void rank_(walk_t& walk, std::span<scored_node_t const> scored) noexcept {
        std::size_t const accepted = walk.results.insert(scored, shape_.expansion_factor);
        std::size_t const queued = walk.stage == nvme_walk_stage_t::seeding_k ? scored.size() : accepted;
        for (std::size_t lane = 0; lane != queued; ++lane)
            if (failed(walk.frontier.insert(scored_node_t(scored[lane])))) return saturate_(walk);
    }

    /** Expands the closest frontier nodes into the free hop slots of @p walk_index while each
     *  lies inside a full pool's radius; with nothing in flight, re-seeds a frontier drained
     *  before the bound fills, or finishes. */
    void advance_(u32_t walk_index) noexcept {
        walk_t& walk = walks_[walk_index];
        if (walk.stage == nvme_walk_stage_t::finished_k) return;
        if (walk.stage == nvme_walk_stage_t::draining_k) {
            if (!in_flight_(walk_index)) finish_walk_(walk_index);
            return;
        }
        row_t const& row = rows_[walk.row_slot];
        std::size_t const expansion_factor = shape_.expansion_factor;
        results_array_t const& bound = row.bound(walk);
        results_array_t const& pool = bound.size() == expansion_factor ? bound : walk.results;
        bool const pool_full = pool.size() == expansion_factor;
        std::span<u32_t> const hop_flights = hop_flights_of_(walk_index);
        for (u32_t slot = 0; slot != hop_flights.size() && !walk.frontier.empty(); ++slot) {
            if (hop_flights[slot]) continue;
            scored_node_t const candidate = walk.frontier.top();
            if (pool_full && candidate.distance > pool.top().distance) break;
            if (pool_full && walk.frontier.capacity() - walk.frontier.size() < parts_[walk.shard].connectivity)
                walk.frontier.erase_below(pool.top());
            walk.frontier.pop();
            ++telemetry_.visited_members;
            u32_t const flight = take_flight_();
            flights_[flight] = flight_t {walk_index, candidate.node_id, slot, 1, nvme_purpose_t::expand_k};
            issue_(flight, nvme_purpose_t::expand_k, flight);
            hop_flights[slot] = 1;
        }
        if (in_flight_(walk_index)) return;
        // A walk only reaches nodes connected to its seeds, so a frontier drained before the
        // bound fills re-seeds.
        if (walk.frontier.empty() && bound.size() < expansion_factor) {
            walk.stage = nvme_walk_stage_t::seeding_k;
            if (seed_(walk_index, 4)) return void(++telemetry_.restarts);
        }
        finish_walk_(walk_index);
    }

    /** Reads the landed neighbors row of @p flight_index and issues a score for every unvisited
     *  neighbor, in the expansion's own hop. */
    void expand_(u32_t walk_index, u32_t flight_index) noexcept {
        walk_t& walk = walks_[walk_index];
        part_t const& part = parts_[walk.shard];
        u32_t const hop = flights_[flight_index].position;
        std::span<u32_t> const hop_flights = hop_flights_of_(walk_index);
        --hop_flights[hop];
        if (walk.stage != nvme_walk_stage_t::draining_k) {
            if (failed(walk.visited.reserve(walk.visited.size() + part.connectivity))) saturate_(walk);
            else {
                auto const* const neighbors = reinterpret_cast<node_id_t const*>(
                    landed_row_(flight_index, nvme_purpose_t::expand_k, flight_index));
                // A row holds as many IDs as its own bytes, at most the part's widest row.
                std::size_t const degree = tier_t::row_length_(part, nvme_purpose_t::expand_k, flights_[flight_index]) /
                                           sizeof(node_id_t);
                for (std::size_t slot = 0; slot != degree; ++slot) {
                    node_id_t const neighbor = neighbors[slot];
                    if (neighbor >= part.nodes || walk.visited.insert(neighbor) == membership_t::present_k) continue;
                    issue_score_(walk_index, neighbor, hop);
                    ++hop_flights[hop];
                }
            }
        }
        release_flight_(flight_index);
    }

    /** Merges @p walk_index into its row, and once every walk of the row finished, starts the
     *  row's resolution. */
    void finish_walk_(u32_t walk_index) noexcept {
        walk_t& walk = walks_[walk_index];
        walk.stage = nvme_walk_stage_t::finished_k;
        u32_t const slot_index = walk.row_slot;
        nvme_row_slot_t& slot = slots_[slot_index];
        assert(slot.stage == nvme_row_stage_t::walking_k && "Only a walking row's walks finish");
        rows_[slot_index].finish_walk(walk);
        if (--slot.walks_left) return;
        slot.stage = nvme_row_stage_t::resolving_k;
        slot.outputs = static_cast<u32_t>(rows_[slot_index].outputs());
        resolve_row_(slot_index);
    }

    /** Issues key reads for the row's written locators within the row's flight budget, and
     *  finishes the row once every one landed. */
    void resolve_row_(u32_t slot_index) noexcept {
        nvme_row_slot_t& slot = slots_[slot_index];
        assert(slot.stage == nvme_row_stage_t::resolving_k && "Only a resolving row reads keys");
        row_t& row = rows_[slot_index];
        std::size_t const budget = parts_.size() * shape_.walk_flights;
        while (slot.next_output != slot.outputs && slot.resolving != budget) {
            auto const locator = row.locator(slot.next_output);
            u32_t const flight = take_flight_();
            flights_[flight] = flight_t {slot_index, locator.node, slot.next_output, 1, nvme_purpose_t::key_k};
            issue_(flight, nvme_purpose_t::key_k, flight);
            ++slot.next_output;
            ++slot.resolving;
        }
        if (slot.resolving) return;
        if (status_t const finished = row.finish(); failed(finished) && succeeded(refusal_)) refusal_ = finished;
        if (!shape_.latencies.empty())
            shape_.latencies[slot.row] = static_cast<u64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - slot.primed)
                    .count());
        slot.stage = nvme_row_stage_t::idle_k;
        --rows_in_flight_;
    }

    engine_t& engine_;
    arena_t& arena_;
    std::span<medium_t const> media_;
    std::span<part_t const> parts_;
    walk_shape_t const& shape_;

    std::span<walk_t> walks_;
    std::span<row_t> rows_;
    std::span<nvme_row_slot_t> slots_;
    std::span<flight_t> flights_;
    std::span<u32_t> free_flights_;
    std::span<u32_t> hop_flights_;
    std::span<io_request_t> backlog_;
    std::span<io_completion_t> completions_;
    std::size_t free_count_ = 0;
    std::size_t backlog_size_ = 0;
    std::size_t outstanding_ = 0;
    std::size_t rows_in_flight_ = 0;
    search_result_t telemetry_;

    /** The read failure that stops this thread. */
    status_t failure_ = status_t::success_k;

    /** The first row that refused, which leaves the other rows standing. */
    status_t refusal_ = status_t::success_k;
};

#pragma endregion Walks

} // namespace unum::usearch::utilities
