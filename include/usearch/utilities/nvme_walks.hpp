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
 *  - @c walk_t, one walk's candidates and admitted set, and @c walk_shape_t, the call's sizes;
 *  - @c walk_payload_t, what an aggregating predicate carries beside each key it gathers;
 *  - @c prime_scoring_, which prepares a row's query once the row primes;
 *  - @c score_staged_, which scores a walk's staged flights, folds and releases them, and ranks the
 *    rest through the walker's @c rank_;
 *  - @c ask_relation_, which asks a row's relation about the landed rows of contending flights.
 *
 *  Nothing here reads a value row, so the file needs no kernel library, and every tier compiles the
 *  same walk.
 */
#pragma once

#include <cassert> // `assert`
#include <cstring> // `std::memset`

#include <array>       // `std::array`
#include <bit>         // `std::bit_cast`
#include <chrono>      // `std::chrono::steady_clock`, for per-row latencies
#include <concepts>    // `std::same_as`
#include <new>         // placement `new`
#include <span>        // `std::span`
#include <string_view> // `std::string_view`
#include <type_traits> // `std::remove_cv_t`

#include "usearch/utilities/types.hpp"
#include "usearch/utilities/nvme_device.hpp"       // `io_request_t`, `io_completion_t`, `block_bytes_of`
#include "usearch/utilities/ram_buffers.hpp"       // `growing_buffer`
#include "usearch/utilities/ram_device.hpp"        // `forkunion_allocator_t`
#include "usearch/utilities/ram_tables.hpp"        // `membership_t`
#include "usearch/utilities/safegraphs_format.hpp" // `safegraphs_read_metadata`, `safegraphs_widened_key`

namespace unum::usearch::utilities {

#pragma region Parts

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

/**
 *  Reads the prefix and the header of the part @p source holds, any reader with
 *  <tt>read_at(offset, bytes)</tt>, into @p header_bytes.
 *
 *  @par Statuses:
 *  - @c malformed_k The prefix cannot be read or carries no monolith's magic.
 *  - @c malformed_k The header cannot be read.
 */
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

/**
 *  Reads the prefix and the header of the part at @p path into @p header_bytes through the medium
 *  a view reads it by: a file or a device, or in SPDK builds a namespace a transport ID names,
 *  each with an optional @c base: token.
 *
 *  @par Statuses:
 *  - @c invalid_argument_k @p path names no location, its base or namespace is unreadable, or the
 *    base is off the medium's block; in SPDK builds also an inactive namespace or an unparsable
 *    transport ID.
 *  - @c unsupported_k A transport ID in a build without SPDK.
 *  - @c malformed_k No monolith starts there, or its header cannot be read.
 */
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

/**
 *  The metadata of the checkpoint at @p path without viewing it: a manifest in a file, or a
 *  monolith wherever @c nvme_read_header reaches one, its header read as a view reads it.
 *
 *  @par Statuses:
 *  - @c invalid_argument_k @p path, or a manifest's first part path, does not parse or names an
 *    unusable base or namespace.
 *  - @c unsupported_k A transport ID in a build without SPDK.
 *  - @c malformed_k A transport ID starting no monolith, or a manifest or header that does not
 *    parse.
 */
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

/** The three queue reservations one walk starts from, at one expansion over parts whose widest row
 *  holds @c connectivity neighbors - derived once with the call's shape and carried in it. */
struct nvme_queue_sizes_t {

    /** How many candidates the walk keeps: the expansion factor. */
    std::size_t results = 0;

    /** The full candidates' unexpanded members plus one expansion's admissions, which shedding past
     *  their radius keeps the walk within; ties at that radius grow it. */
    std::size_t frontier = 0;

    /** One row per candidate - the first size a re-table doubles from. */
    std::size_t visited = 0;
};

/** The queue sizes of a walk @p expansion_factor wide over rows of at most @p connectivity
 *  neighbors. */
constexpr nvme_queue_sizes_t nvme_queue_sizes(std::size_t expansion_factor, std::size_t connectivity) noexcept {
    return {expansion_factor, expansion_factor + connectivity, expansion_factor * connectivity};
}

/** The most neighbors any row of @p parts holds - what sizes every walk over all of them. */
template <typename part_type_>
std::size_t nvme_widest_row(std::span<part_type_ const> parts) noexcept {
    std::size_t widest = 0;
    for (part_type_ const& part : parts) widest = max_of<std::size_t>(widest, part.connectivity);
    return widest;
}

/** Where a walk stands between its reads. */
enum class nvme_walk_stage_t : u8_t {

    /** Finished, its candidates merged into its row, or never primed. */
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

/** A node one walk scored, which an aggregating predicate carries beside its key until the walker
 *  settles the batch it landed in; flat, so a 32-bit node and a 64-bit distance fit 16 bytes. */
template <typename node_id_type_, typename distance_type_>
struct nvme_walk_payload {

    /** The walk that scored it, positional among its thread's walks. */
    u32_t walk = 0;

    /** The node scored. */
    node_id_type_ node_id {};

    /** Its distance. */
    distance_type_ distance {};
};

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

/**
 *  Claims @p executor, sizes the arena of every thread at @p scratch_bytes and every engine at
 *  @p buffers staging buffers of @p buffer_bytes, and attaches @p media. Nothing rewinds the
 *  arenas again before the call ends, so whatever a call carves lasts all of it.
 *
 *  @par Statuses:
 *  - @c contended_k Another call holds @p executor, or its pool is running another dispatch.
 *  - @c uninitialized_k The executor holds no threads, or its pool has no workers.
 *  - @c overflow_k A staging region's byte count overflows, or passes what Windows registers.
 *  - @c config_mismatch_k A medium the engines cannot read.
 */
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
 *  @tparam cache_hints_type_ The line hints streaming a landed row toward the core while its quad
 *      fills.
 */
template <typename tier_type_, typename row_type_, typename cache_hints_type_ = dummy_cache_hints_t>
class row_walker {
    friend tier_type_;

    using tier_t = tier_type_;
    using engine_t = typename tier_t::engine_t;
    using arena_t = typename tier_t::arena_t;
    using medium_t = typename tier_t::medium_t;
    using part_t = typename tier_t::part_t;
    using node_id_t = typename tier_t::node_id_t;
    template <typename word_type_>
    using atomic_ref = typename tier_t::template atomic_ref<word_type_>;
    using flight_t = nvme_flight<node_id_t>;
    using walk_t = typename tier_t::walk_t;
    using walk_shape_t = typename tier_t::walk_shape_t;
    using search_result_t = typename tier_t::search_result_t;
    using scratch_allocator_t = typename tier_t::scratch_allocator_t;
    using results_array_t = typename tier_t::results_array_t;
    using scored_node_t = typename results_array_t::value_type;
    using walk_payload_t = typename tier_t::walk_payload_t;

  public:
    using row_t = row_type_;
    using key_t = typename row_t::key_t;
    using inputs_t = typename row_t::inputs_t;

    /** The walker of executor thread @p thread, whose rows are built from @p inputs, and which asks
     *  that thread's predicate among them. */
    row_walker(engine_t& engine, arena_t& arena, std::span<medium_t const> media, std::span<part_t const> parts,
               walk_shape_t const& shape, inputs_t const& inputs, std::size_t thread, std::size_t line_bytes) noexcept
        : engine_(engine), arena_(arena), media_(media), parts_(parts), shape_(shape), inputs_(inputs), thread_(thread),
          prefetcher_ {line_bytes} {}

    row_walker(row_walker const&) = delete;
    row_walker& operator=(row_walker const&) = delete;

    ~row_walker() noexcept {
        for (walk_t& walk : walks_) walk.~walk_t();
        for (row_t& row : rows_) row.~row_t();
    }

    /** Carves every walk, row and flight of the thread, and reserves an aggregating predicate for
     *  the widest batch it can gather: every flight of the thread landing at once. */
    status_t reserve() noexcept {
        std::size_t const parts = parts_.size(), rows = shape_.rows_per_thread;
        std::size_t const walks = rows * parts, flights = walks * shape_.walk_flights;
        walk_t* const walk_slots = carve_<walk_t>(walks);
        row_t* const row_slots = carve_<row_t>(rows);
        nvme_row_slot_t* const slot_states = carve_<nvme_row_slot_t>(rows);
        flight_t* const flight_states = carve_<flight_t>(flights);
        u32_t* const free_flights = carve_<u32_t>(flights);
        u32_t* const hop_flights = carve_<u32_t>(walks * shape_.hop_slots);
        io_request_t* const backlog = carve_<io_request_t>(shape_.buffers);
        io_completion_t* const completions = carve_<io_completion_t>(shape_.buffers);
        if (!walk_slots || !row_slots || !slot_states || !flight_states || !free_flights || !hop_flights || !backlog ||
            !completions)
            return status_t::bad_alloc_k;

        scratch_allocator_t const allocator {arena_};
        nvme_queue_sizes_t const queue_sizes = shape_.queue_sizes;
        for (std::size_t walk = 0; walk != walks; ++walk) {
            walk_t& constructed = *new (walk_slots + walk) walk_t {allocator};
            walks_ = {walk_slots, walk + 1};
            if (failed(constructed.results.reserve(queue_sizes.results)) ||
                failed(constructed.frontier.reserve(queue_sizes.frontier)) ||
                failed(constructed.visited.reserve(queue_sizes.visited)) ||
                (row_t::admitting_k && failed(constructed.admitted.reserve(queue_sizes.results))))
                return status_t::bad_alloc_k;
        }
        for (std::size_t row = 0; row != rows; ++row) {
            row_t& constructed = *new (row_slots + row) row_t {inputs_, allocator};
            rows_ = {row_slots, row + 1};
            if (status_t const reserved = constructed.reserve(parts, nvme_widest_row(parts_)); failed(reserved))
                return reserved;
        }
        for (std::size_t row = 0; row != rows; ++row) new (slot_states + row) nvme_row_slot_t {};
        for (std::size_t flight = 0; flight != flights; ++flight)
            free_flights[flight] = static_cast<u32_t>(flights - 1 - flight);
        slots_ = {slot_states, rows};
        flights_ = {flight_states, flights};
        free_flights_ = {free_flights, flights};
        hop_flights_ = {hop_flights, walks * shape_.hop_slots};
        free_count_ = flights;
        backlog_ = {backlog, shape_.buffers};
        completions_ = {completions, shape_.buffers};
        if constexpr (row_t::admitting_k)
            return reserve_predicate(thread_predicate(*inputs_.predicate, thread_), flights);
        return status_t::success_k;
    }

    /** Primes rows until the batch runs out, keeping every slot's walks in flight, and drains the
     *  engine before it returns, whatever happened; a failure on any thread stops this one too.
     *  Answers the first row that refused, since a failure is published to every thread
     *  instead. */
    status_t drive_walks() noexcept {
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
                    if (route_completion_(completion)) completions_[advancing++] = completion;
                // Walks advance once the whole batch landed and its admissions settled, so their
                // next hops weigh every row that did.
                settle_admissions_();
                if (succeeded(failure_))
                    for (io_completion_t const& completion : completions_.first(advancing))
                        advance_walk_(static_cast<u32_t>(completion.context));
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
        atomic_ref<status_t>(*shape_.failure)
            .compare_exchange_strong(unset, failure, std::memory_order_relaxed, std::memory_order_relaxed);
    }

    /**
     *  @brief Walks every row of a batch over every part in @p parts, the rows' walks interleaved
     *      on every thread of @p executor, each row built from @p inputs as @p row_type_ says.
     *
     *  Runs inside a call @ref nvme_begin_call began at @p shape 's sizes, and sums the threads'
     *  telemetry; @p counts are the rows' own, summed into the result's @c count. @p latencies
     *  takes one duration per row, or is empty. Every thread streams its landed rows through
     *  @p prefetcher.
     */
    template <typename executor_type_>
    static expected<search_result_t> run_walks(std::span<medium_t const> media, std::span<part_t const> parts,
                                               executor_type_& executor, walk_shape_t shape,
                                               typename row_t::inputs_t const& inputs,
                                               std::span<std::size_t const> counts, std::span<u64_t> latencies,
                                               std::size_t line_bytes = default_alignment_k) noexcept {
        search_result_t merged;
        if (!latencies.empty() && latencies.size() != shape.rows_count) return {merged, status_t::config_mismatch_k};
        alignas(default_alignment_k) std::size_t cursor = 0;
        status_t failure = status_t::success_k;
        shape.cursor = &cursor;
        shape.failure = &failure;
        shape.latencies = latencies;
        status_t refusal = status_t::success_k;
        auto add = [](std::size_t& into, std::size_t value) noexcept {
            atomic_ref<std::size_t>(into).add(value, std::memory_order_relaxed);
        };
        auto add_reads = [&](nvme_reads_t& into, nvme_reads_t const& reads) noexcept {
            add(into.count, reads.count), add(into.bytes, reads.bytes);
        };
        auto walk_thread = [&](std::size_t thread, std::size_t) noexcept {
            row_walker walker {
                executor.engine(thread), executor.arena(thread), media, parts, shape, inputs, thread, line_bytes};
            status_t const reserved = walker.reserve();
            if (failed(reserved)) walker.fail(reserved);
            status_t const refused = walker.drive_walks();
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
                atomic_ref<status_t>(refusal).compare_exchange_strong(unset, refused, std::memory_order_relaxed,
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
        return atomic_ref<status_t>(*shape_.failure).load(std::memory_order_relaxed);
    }

    bool exhausted_() const noexcept {
        return atomic_ref<std::size_t>(*shape_.cursor).load(std::memory_order_relaxed) >= shape_.rows_count;
    }

    /** Hop slots a walk may hold now: its own share, and once the batch ran out, the shares of
     *  the row slots that went idle; the shares only grow after that, so the free list never runs
     *  dry. */
    std::size_t live_hops_() const noexcept {
        std::size_t const rows = shape_.rows_per_thread;
        if (rows_in_flight_ == rows || !exhausted_()) return shape_.hops_per_walk;
        std::size_t const inherited = (rows - rows_in_flight_) * shape_.walk_flights /
                                      (rows_in_flight_ * (nvme_widest_row(parts_) + 1));
        return min_of(shape_.hop_slots, shape_.hops_per_walk + inherited);
    }

    u32_t take_flight_() noexcept { return free_flights_[--free_count_]; }
    void release_flight_(u32_t flight) noexcept { free_flights_[free_count_++] = flight; }

    /** The flights each hop slot of @p walk_index waits on; a zero slot is free. */
    std::span<u32_t> hop_flights_of_(u32_t walk_index) noexcept {
        return hop_flights_.subspan(walk_index * shape_.hop_slots, shape_.hop_slots);
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
        return flight.purpose == nvme_purpose_t::key_k ? rows_[flight.issuer].output_locator(flight.position).shard
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
    void issue_row_read_(u32_t flight_index, nvme_purpose_t purpose, io_buffer_id_t buffer) noexcept {
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
            std::size_t const row = atomic_ref<std::size_t>(*shape_.cursor).fetch_add(1, std::memory_order_relaxed);
            if (row >= shape_.rows_count) return;
            slot = nvme_row_slot_t {static_cast<u32_t>(parts), 0, 0, 0, nvme_row_stage_t::walking_k, row, {}};
            if (!shape_.latencies.empty()) slot.primed = std::chrono::steady_clock::now();
            ++rows_in_flight_;
            row_t& primed = rows_[slot_index];
            primed.prime_row(row);
            tier_t::prime_scoring_(shape_, primed);
            for (u32_t shard = 0; shard != parts; ++shard) {
                u32_t const walk_index = slot_index * static_cast<u32_t>(parts) + shard;
                walk_t& walk = walks_[walk_index];
                walk.results.clear();
                walk.admitted.clear();
                walk.frontier.clear();
                walk.visited.clear();
                walk.staged.clear();
                walk.random_generator = seeded_stream(u64_t(row) * parts + shard + 1);
                walk.shard = shard;
                walk.row_slot = slot_index;
                walk.stage = nvme_walk_stage_t::seeding_k;
                std::span<u32_t> const hop_flights = hop_flights_of_(walk_index);
                std::memset(hop_flights.data(), 0, hop_flights.size_bytes());
                std::size_t const landmarks = round_up_to_multiple<std::size_t>(parts_[shard].connectivity, 4);
                if (!seed_landmarks_(walk_index, landmarks)) finish_walk_(walk_index);
            }
        }
    }

    /** Issues up to @p count fresh landmarks of walk @p walk_index as one hop, answering how
     *  many. A walk seeds only with nothing in flight, so the hop takes slot zero. */
    std::size_t seed_landmarks_(u32_t walk_index, std::size_t count) noexcept {
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
        issue_row_read_(flight, nvme_purpose_t::score_k, flight);
        if constexpr (row_t::key_reads_k == nvme_key_reads_t::scored_k)
            issue_row_read_(flight, nvme_purpose_t::key_k, key_buffer_(flight));
    }

    /** Routes one completion to its flight, answering whether its issuer is a walk the batch's
     *  end advances; a read short of its row fails the thread. */
    bool route_completion_(io_completion_t const& completion) noexcept {
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
            prefetcher_.prefetch(landed_row_(flight_index, nvme_purpose_t::score_k, flight_index),
                                 row_bytes_(flight, nvme_purpose_t::score_k));
            walk_t& walk = walks_[flight.issuer];
            std::span<u32_t> const hop_flights = hop_flights_of_(flight.issuer);
            [[maybe_unused]] fu::status_t const queued = walk.staged.push_back(flight_index);
            // A quad scores once full or once a hop's last row lands, whichever hops its lanes
            // came from.
            if (--hop_flights[flight.position] && !walk.staged.full()) return true;
            score_staged_flights_(walk);
            // Landmarks take hop slot zero, so seeding ends once that slot drains.
            if (walk.stage == nvme_walk_stage_t::seeding_k && !hop_flights[0])
                walk.stage = nvme_walk_stage_t::expanding_k;
            return true;
        }
        case nvme_purpose_t::expand_k: expand_landed_row_(flight.issuer, flight_index); return true;
        case nvme_purpose_t::key_k: {
            u32_t const slot_index = flight.issuer;
            byte_t const* const stored = landed_row_(flight_index, nvme_purpose_t::key_k, flight_index);
            dtype_t const stored_dtype = parts_[shard_of_(flight)].key_dtype;
            rows_[slot_index].resolve_output_key(flight.position, safegraphs_widened_key<key_t>(stored, stored_dtype));
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
     *  through @c rank_. */
    void score_staged_flights_(walk_t& walk) noexcept {
        if (!walk.staged.size()) return;
        if (walk.stage == nvme_walk_stage_t::draining_k) {
            for (u32_t const flight : walk.staged) release_flight_(flight);
            return walk.staged.clear();
        }
        tier_t::score_staged_(*this, walk);
    }

    /** Merges the ascending @p scored rows of @p walk into its candidates, and queues them for
     *  expansion: every landmark, as the in-memory walk seeds the frontier, and a neighbor only
     *  when the candidates took it. */
    [[gnu::always_inline]] void rank_(walk_t& walk, std::span<scored_node_t const> scored) noexcept {
        std::size_t const accepted = walk.results.insert(scored, shape_.expansion_factor);
        std::size_t const queued = walk.stage == nvme_walk_stage_t::seeding_k ? scored.size() : accepted;
        for (std::size_t lane = 0; lane != queued; ++lane)
            if (failed(walk.frontier.insert(scored_node_t(scored[lane])))) return saturate_(walk);
    }

    /** Admits the nodes of @p scored, scored from the staged flights of @p walk in lane order, that
     *  could still enter its admitted set and that the row's relation and the thread's predicate
     *  both admit: at once under an immediate predicate, at the batch's settle under an aggregating
     *  one. */
    void admit_scored_(walk_t& walk, std::span<scored_node_t const> scored) noexcept {
        with_static_extent<quad_lanes_k>(scored.size(), [&]<std::size_t extent_>() noexcept {
            admit_contenders_(walk, std::span<scored_node_t const, extent_>(scored.data(), extent_));
        });
    }

    /** Asks about the lanes of @p scored that could still enter the admitted set of @p walk: all
     *  while it is short, and once full only those closer than its furthest member. */
    template <std::size_t extent_>
    void admit_contenders_(walk_t& walk, std::span<scored_node_t const, extent_> scored) noexcept {
        results_array_t const& admitted = walk.admitted;
        bool const admitted_full = admitted.size() == shape_.expansion_factor;
        std::array<u32_t, extent_> flights;
        std::array<scored_node_t, extent_> contenders;
        std::size_t contending = 0;
        for (std::size_t lane = 0; lane != extent_; ++lane) {
            if (admitted_full && !(scored[lane].distance < admitted.top().distance)) continue;
            flights[contending] = walk.staged[lane];
            contenders[contending] = scored[lane];
            ++contending;
        }
        with_static_extent<extent_>(contending, [&]<std::size_t contending_extent_>() noexcept {
            relate_contenders_(
                walk, std::span<u32_t const, contending_extent_>(flights.data(), contending_extent_),
                std::span<scored_node_t const, contending_extent_>(contenders.data(), contending_extent_));
        });
    }

    /** Asks the row's relation about the landed rows of the contending @p flights, then the
     *  thread's predicate about the keys of those it kept. */
    template <std::size_t extent_>
    void relate_contenders_(walk_t& walk, std::span<u32_t const, extent_> flights,
                            std::span<scored_node_t const, extent_> contenders) noexcept {
        std::array<admission_t, extent_> related;
        tier_t::ask_relation_(*this, walk, flights, std::span<admission_t, extent_>(related));
        u32_t const walk_index = static_cast<u32_t>(&walk - walks_.data());
        std::array<u32_t, extent_> kept_flights;
        std::array<walk_payload_t, extent_> payloads;
        std::size_t kept = 0;
        for (std::size_t lane = 0; lane != extent_; ++lane) {
            if (related[lane] != admission_t::admit_k) continue;
            kept_flights[kept] = flights[lane];
            payloads[kept] = walk_payload_t {walk_index, contenders[lane].node_id, contenders[lane].distance};
            ++kept;
        }
        with_static_extent<extent_>(kept, [&]<std::size_t kept_extent_>() noexcept {
            ask_predicate_(thread_predicate(*inputs_.predicate, thread_),
                           std::span<u32_t const, kept_extent_>(kept_flights.data(), kept_extent_),
                           std::span<walk_payload_t const, kept_extent_>(payloads.data(), kept_extent_));
        });
    }

    /** Without a predicate every related node is admitted, and no key landed to ask about. */
    template <typename predicate_type_, std::size_t extent_>
        requires std::same_as<std::remove_cv_t<predicate_type_>, dummy_predicate_t>
    void ask_predicate_(predicate_type_&, std::span<u32_t const, extent_>,
                        std::span<walk_payload_t const, extent_> payloads) noexcept {
        for (walk_payload_t const& payload : payloads) admit_payload_(payload);
    }

    /** Asks @p predicate about the keys landed beside @p flights, each carrying its payload. */
    template <typename predicate_type_, std::size_t extent_>
    void ask_predicate_(predicate_type_& predicate, std::span<u32_t const, extent_> flights,
                        std::span<walk_payload_t const, extent_> payloads) noexcept {
        std::array<key_t, extent_> keys;
        for (std::size_t lane = 0; lane != extent_; ++lane) keys[lane] = landed_key_(flights[lane]);
        ask_quad(predicate, std::span<key_t const, extent_>(keys), payloads,
                 [this](walk_payload_t const& payload) noexcept { admit_payload_(payload); });
    }

    /** Answers what the thread's predicate gathered over a completion batch, so every walk advances
     *  on its whole admitted set. */
    void settle_admissions_() noexcept {
        if constexpr (row_t::admitting_k)
            settle_predicate(thread_predicate(*inputs_.predicate, thread_),
                             [this](walk_payload_t const& payload) noexcept { admit_payload_(payload); });
    }

    /** Inserts an admitted node into the admitted set of the walk that scored it. */
    void admit_payload_(walk_payload_t const& payload) noexcept {
        [[maybe_unused]] bool const inserted = walks_[payload.walk].admitted.insert(
            scored_node_t {payload.node_id, payload.distance}, shape_.expansion_factor);
    }

    /** The key landed beside the value of the keyed score @p flight_index. */
    key_t landed_key_(u32_t flight_index) noexcept {
        return safegraphs_widened_key<key_t>(
            landed_row_(flight_index, nvme_purpose_t::key_k, key_buffer_(flight_index)),
            parts_[shard_of_(flights_[flight_index])].key_dtype);
    }

    /** Expands the closest frontier nodes into the free hop slots of @p walk_index while each lies
     *  inside the full stopping candidates' radius; with nothing in flight, re-seeds a frontier
     *  drained before the bound fills, or finishes. */
    void advance_walk_(u32_t walk_index) noexcept {
        walk_t& walk = walks_[walk_index];
        if (walk.stage == nvme_walk_stage_t::finished_k) return;
        if (walk.stage == nvme_walk_stage_t::draining_k) {
            if (!in_flight_(walk_index)) finish_walk_(walk_index);
            return;
        }
        row_t const& row = rows_[walk.row_slot];
        std::size_t const expansion_factor = shape_.expansion_factor;
        results_array_t const& stopping = row.stopping_candidates(walk);
        bool const stopping_full = stopping.size() == expansion_factor;
        std::span<u32_t> const hop_flights = hop_flights_of_(walk_index).first(live_hops_());
        for (u32_t slot = 0; slot != hop_flights.size() && !walk.frontier.empty(); ++slot) {
            if (hop_flights[slot]) continue;
            scored_node_t const candidate = walk.frontier.top();
            if (stopping_full && candidate.distance > stopping.top().distance) break;
            if (stopping_full && walk.frontier.capacity() - walk.frontier.size() < parts_[walk.shard].connectivity)
                walk.frontier.erase_below(stopping.top());
            walk.frontier.pop();
            ++telemetry_.visited_members;
            u32_t const flight = take_flight_();
            flights_[flight] = flight_t {walk_index, candidate.node_id, slot, 1, nvme_purpose_t::expand_k};
            issue_row_read_(flight, nvme_purpose_t::expand_k, flight);
            hop_flights[slot] = 1;
        }
        if (in_flight_(walk_index)) return;
        // A walk only reaches nodes connected to its seeds, so a frontier drained before the
        // stopping candidates fill re-seeds.
        if (walk.frontier.empty() && stopping.size() < expansion_factor) {
            walk.stage = nvme_walk_stage_t::seeding_k;
            if (seed_landmarks_(walk_index, 4)) return void(++telemetry_.restarts);
        }
        finish_walk_(walk_index);
    }

    /** Reads the landed neighbors row of @p flight_index and issues a score for every unvisited
     *  neighbor, in the expansion's own hop. */
    void expand_landed_row_(u32_t walk_index, u32_t flight_index) noexcept {
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
        slot.outputs = static_cast<u32_t>(rows_[slot_index].output_count());
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
            auto const locator = row.output_locator(slot.next_output);
            u32_t const flight = take_flight_();
            flights_[flight] = flight_t {slot_index, locator.node, slot.next_output, 1, nvme_purpose_t::key_k};
            issue_row_read_(flight, nvme_purpose_t::key_k, flight);
            ++slot.next_output;
            ++slot.resolving;
        }
        if (slot.resolving) return;
        if (status_t const finished = row.finish_row(); failed(finished) && succeeded(refusal_)) refusal_ = finished;
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
    inputs_t const& inputs_;

    /** The executor thread this walker runs on, whose predicate it asks. */
    std::size_t thread_ = 0;

    /** Streams each landed row toward the core while the rest of its quad is still in flight. */
    object_prefetcher<cache_hints_type_> prefetcher_;

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
