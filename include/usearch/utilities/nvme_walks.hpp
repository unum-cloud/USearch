/**
 *  @file include/usearch/utilities/nvme_walks.hpp
 *  @author Ash Vardanian
 *  @date July 22, 2026
 *  @brief The storage walk every NVMe tier drives: the parts' vocabulary, one thread's walks as a
 *      pure state machine of reads, and what a SAMPLE call reads beside its scores.
 *
 *  A walk never performs I/O itself: it only emits read requests and consumes completions, and the
 *  engines of @c nvme_device.hpp submit them. What a fetched value row means belongs to the tier
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
 *  - @c prime_scoring_, which prepares a query once it primes;
 *  - @c score_staged_, which scores a walk's staged flights, folds and releases them, and ranks the
 *    rest through the walker's @c rank_;
 *  - @c ask_relation_, which asks a query's relation about the fetched rows of contending flights.
 *
 *  Nothing here reads a value row, so the file needs no kernel library, and every tier compiles the
 *  same walk.
 */
#pragma once

#include <cassert> // `assert`
#include <cstring> // `std::memset`, `std::memcpy`

#include <array>       // `std::array`
#include <bit>         // `std::bit_cast`, `std::popcount`, `std::countr_zero`
#include <chrono>      // `std::chrono::steady_clock`, for per-query latencies
#include <concepts>    // `std::same_as`
#include <functional>  // `std::less`, `std::hash` - each key's listed entries in order, and the keys' table
#include <limits>      // `std::numeric_limits`
#include <new>         // placement `new`
#include <span>        // `std::span`
#include <string_view> // `std::string_view`
#include <type_traits> // `std::remove_cv_t`

#include "usearch/utilities/types.hpp"
#include "usearch/utilities/nvme_device.hpp"       // `io_request_t`, `io_completion_t`, `block_bytes_of`
#include "usearch/utilities/ram_buffers.hpp"       // `growing_buffer`
#include "usearch/utilities/ram_device.hpp"        // `forkunion_allocator_t`
#include "usearch/utilities/ram_sorts.hpp"         // `partial_sort` - the order located reads merge in
#include "usearch/utilities/ram_tables.hpp"        // `chunked_multimap`, `fingerprint_multimap` - listed and held keys
#include "usearch/utilities/ram_walks.hpp"         // `cursor_walk` - a parts cursor's order
#include "usearch/utilities/safegraphs_format.hpp" // `safegraphs_read_metadata`, `safegraphs_widened_key`
#include "usearch/utilities/sample.hpp"            // `sample_plan` - what a SAMPLE call over parts draws through

namespace unum::usearch::utilities {

#pragma region Parts

/** Maps a walked node ID of one shard to its @c shard_node, as an ID resolver and as an ID-to-key
 *  table. */
template <typename node_id_type_>
struct nvme_shard_locators {

    /** The shard every resolved ID belongs to. */
    u32_t shard = 0;

    shard_node<node_id_type_> operator()(node_id_type_ node) const noexcept { return {shard, node}; }
    shard_node<node_id_type_> operator[](node_id_type_ node) const noexcept { return {shard, node}; }
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
 *  arriving in @p buffer, its completion carrying @p context back. */
inline io_request_t nvme_covering_read(u64_t begin, u64_t bytes, u64_t block_bytes, io_medium_id_t medium,
                                       io_buffer_id_t buffer, u64_t context) noexcept {
    u64_t const first = begin - begin % block_bytes;
    u64_t const bound = round_up_to_multiple<u64_t>(begin + bytes, block_bytes);
    assert(bound - first <= io_max_length_k && "A covering read must fit the length a completion counts");
    return {first, static_cast<u32_t>(bound - first), medium, buffer, context};
}

/** The bytes one blocking round reads from one part: where they start past the part's byte zero,
 *  how many, and which attached medium holds the part. */
struct nvme_extent_t {

    /** Byte offset past the part's start. */
    u64_t begin = 0;

    /** Bytes wanted, whose covering read must fit one staging buffer. */
    u32_t length = 0;

    /** The attached medium holding the part. */
    io_medium_id_t medium = 0;
};

static_assert(sizeof(nvme_extent_t) == 16, "An extent is two words, so a round's extents pack densely");

/**
 *  @brief Reads every extent of @p extents through @p engine in one round and waits until all
 *      arrive, extent @c i in staging buffer @c i and @p bytes[i] pointing at its first byte there.
 *
 *  The engine holds @p media attached, is at least as deep as @p extents is long, and its buffers
 *  cover every extent's read.
 *
 *  @par Statuses:
 *  - @c overflow_k The engine holds fewer reads than the round issues.
 *  - @c io_k A read failed or ended before the last byte of its extent.
 */
template <typename engine_type_>
status_t nvme_read_round(engine_type_& engine, std::span<typename engine_type_::medium_t const> media,
                         std::span<nvme_extent_t const> extents, std::span<byte_t const*> bytes) noexcept {
    assert(bytes.size() >= extents.size() && "Every extent lands in a pointer of its own");
    status_t status = status_t::success_k;
    std::size_t issued = 0;
    for (; issued != extents.size(); ++issued) {
        nvme_extent_t const& extent = extents[issued];
        typename engine_type_::medium_t const& medium = media[extent.medium];
        io_request_t const request = nvme_covering_read(byte_base_of(medium) + extent.begin, extent.length,
                                                        block_bytes_of(medium), extent.medium,
                                                        static_cast<io_buffer_id_t>(issued), issued);
        if (failed(status = engine.enqueue_read(request))) break;
    }
    // Every enqueued read is polled past a failure, as its buffer serves nothing else till then.
    for (std::size_t arrived = 0; arrived != issued;) {
        if (status_t const submitted = engine.submit(io_wait_t::blocking_k); failed(submitted)) return submitted;
        std::array<io_completion_t, 1> completion;
        for (; engine.poll_completions(completion); ++arrived) {
            std::size_t const position = static_cast<std::size_t>(completion[0].context);
            nvme_extent_t const& extent = extents[position];
            typename engine_type_::medium_t const& medium = media[extent.medium];
            u64_t const skipped = (byte_base_of(medium) + extent.begin) % block_bytes_of(medium);
            if (completion[0].result < 0 || static_cast<u64_t>(completion[0].result) < skipped + extent.length)
                status = status_t::io_k;
            bytes[position] = engine.buffer(completion[0].buffer) + skipped;
        }
    }
    return status;
}

/**
 *  @brief Reads the prefix and the header of the part @p source holds, any reader with
 *  <tt>read_at(offset, bytes)</tt>, into @p header_bytes.
 *
 *  @par Statuses:
 *  - @c malformed_k The prefix cannot be read or carries no monolith's magic.
 *  - @c malformed_k The header cannot be read, or claims more bytes than the medium holds.
 */
template <typename source_type_, typename allocator_type_>
    requires requires(source_type_ const& source, bytes_span_t bytes) { source.read_at(0, bytes); }
status_t nvme_read_header(source_type_ const& source, growing_buffer<byte_t, allocator_type_>& header_bytes) noexcept {
    std::array<byte_t, sizeof(safegraphs_prefix_t)> prefix_bytes {};
    if (status_t const read = source.read_at(0, prefix_bytes); failed(read)) return status_t::malformed_k;
    safegraphs_prefix_t const prefix = std::bit_cast<safegraphs_prefix_t>(prefix_bytes);
    if (prefix.magic != safegraphs_prefix_t::magic_k) return status_t::malformed_k;
    std::size_t const header_length = static_cast<std::size_t>(little_endian(prefix.header_length));
    // A damaged length is checked against the medium before it sizes an allocation.
    if constexpr (requires { medium_bytes_of(source); })
        if (header_length > medium_bytes_of(source) - sizeof(prefix)) return status_t::malformed_k;
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
template <typename wait_type_, typename allocator_type_>
status_t nvme_read_header(std::string_view path, growing_buffer<byte_t, allocator_type_>& header_bytes) noexcept {
#if USEARCH_WITH_LINUX_SPDK
    if (names_transport_id(path)) {
        auto [name_space, open_status] = spdk_medium_t::open<wait_type_>(path);
        if (failed(open_status)) return open_status;
        status_t const read = nvme_read_header(name_space, header_bytes);
        name_space.template release<wait_type_>();
        return read;
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
template <typename wait_type_>
expected<safegraphs_metadata_t> nvme_read_metadata(std::string_view path) noexcept {
    growing_buffer<byte_t, forkunion_allocator_t> header_bytes {global_allocator};
    status_t const read = nvme_read_header<wait_type_>(path, header_bytes);
    if (read == status_t::malformed_k && !names_transport_id(path)) return safegraphs_read_manifest(path);
    if (failed(read)) return {safegraphs_metadata_t {}, read};
    return safegraphs_read_metadata(to_span(header_bytes));
}

#if USEARCH_OS_WINDOWS_

/** Whether @p path names a directory, which a relocation appends a part's file name to. */
inline bool nvme_is_directory_(char const* path) noexcept {
    DWORD const attributes = ::GetFileAttributesA(path);
    return attributes != INVALID_FILE_ATTRIBUTES && (attributes & FILE_ATTRIBUTE_DIRECTORY);
}

#else

/** Whether @p path names a directory, which a relocation appends a part's file name to. */
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

    /** Only the outputs a query writes resolve into keys, one read each, once its walks
     *  finished. */
    outputs_k,

    /** Every scored entry's key arrives beside its value, for a fold or a rescoring that reads
     *  keys. */
    scored_k,
};

/**
 *  @brief The key table an NVMe view holds in RAM: per part, a @ref fingerprint_multimap from every
 *      key's 32-bit fingerprint to the nodes stored under it, sized once for the part's rows.
 *
 *  A probe answers candidates in RAM, and distinct keys may share a fingerprint, so each candidate
 *  is confirmed by one read of its key row. The parts' key columns fill the table once, from any
 *  number of threads at a time.
 */
template <typename key_type_, typename node_id_type_, template <typename> class atomic_ref_template_>
class nvme_resident_keys {
  public:
    using key_t = key_type_;
    using node_id_t = node_id_type_;
    using map_t = fingerprint_multimap<node_id_t, forkunion_allocator_t, atomic_ref_template_>;

    explicit nvme_resident_keys(forkunion_allocator_t allocator) noexcept : parts_(allocator), allocator_(allocator) {}

    /** The 32-bit fingerprint @p key is filed under. */
    static u32_t fingerprint(key_t const& key) noexcept {
        return static_cast<u32_t>(mix_bits(static_cast<u64_t>(std::hash<key_t> {}(key))) >> 32);
    }

    /** Bytes the tables of @p parts hold once reserved. */
    template <typename part_type_>
    static std::size_t memory_usage_for(std::span<part_type_ const> parts) noexcept {
        std::size_t bytes = 0;
        for (part_type_ const& part : parts) bytes += map_t::reserved_bytes(part.nodes);
        return bytes;
    }

    /**
     *  @brief Reserves an empty table for every part of @p parts, dropping any held before.
     *
     *  @par Statuses:
     *  - @c bad_alloc_k A table could not be allocated.
     */
    template <typename part_type_>
    status_t reserve(std::span<part_type_ const> parts) noexcept {
        parts_.clear();
        if (failed(parts_.reserve_exactly(parts.size()))) return status_t::bad_alloc_k;
        for (part_type_ const& part : parts) {
            parts_.push_back_reserved(map_t {allocator_});
            if (failed(parts_.back().reserve(part.nodes))) return parts_.clear(), status_t::bad_alloc_k;
        }
        return status_t::success_k;
    }

    /** Files @p node of part @p part under the fingerprint of @p key, while other threads file. */
    void insert(u32_t part, key_t const& key, node_id_t node) noexcept { parts_[part].insert(fingerprint(key), node); }

    /**
     *  @brief Invokes @p callback for every node filed in part @p part under @p key 's fingerprint.
     *  @return The slots probed.
     */
    template <typename callback_type_>
    std::size_t find_each(u32_t part, key_t const& key, callback_type_&& callback) const noexcept {
        return parts_[part].find_each(fingerprint(key), callback);
    }

    /** Whether no table is held. */
    bool empty() const noexcept { return parts_.empty(); }

    /** Drops every table. */
    void reset() noexcept { parts_.reset(); }

    /** Bytes the tables hold. */
    std::size_t memory_usage() const noexcept {
        std::size_t bytes = parts_.memory_usage();
        for (map_t const& held : parts_) bytes += held.memory_usage();
        return bytes;
    }

  private:
    growing_buffer<map_t, forkunion_allocator_t> parts_;
    forkunion_allocator_t allocator_;
};

#pragma endregion Parts

#pragma region Walks

/** A hinted landmark of one walk: the node its hint's fingerprint found, and which hint of the
 *  walk's query it answers. */
template <typename node_id_type_>
struct nvme_seed {
    node_id_type_ node {};
    u32_t hint = 0;
};

static_assert(sizeof(nvme_seed<u32_t>) == 8, "A seed is an 8-byte hot entry");

/** The three queue reservations one walk starts from, at one expansion over parts whose widest row
 *  holds @c connectivity neighbors - derived once with the call's shape and carried in it. */
struct nvme_queue_sizes_t {

    /** How many candidates the walk keeps: the expansion factor. */
    std::size_t results = 0;

    /** The full candidates' unexpanded members plus one expansion's admissions, which shedding past
     *  their radius keeps the walk within; ties at that radius grow it. */
    std::size_t frontier = 0;

    /** One entry per candidate - the first size a re-table doubles from. */
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

/** The landmarks one walk seeds over nodes of @p connectivity neighbors: that many, in whole
 *  quads, which also bounds the hints it takes. */
constexpr std::size_t nvme_landmarks(std::size_t connectivity) noexcept {
    return round_up_to_multiple<std::size_t>(connectivity, quad_lanes_k);
}

/** The reads one walk's first hop keeps in flight over nodes of at most @p connectivity neighbors:
 *  its landmarks, or one expansion and every neighbor of it, whichever is more. */
constexpr std::size_t nvme_first_hop(std::size_t connectivity) noexcept {
    return max_of<std::size_t>(nvme_landmarks(connectivity), connectivity + 1);
}

/** Where a walk stands between its reads. */
enum class nvme_walk_stage_t : u8_t {

    /** Finished, its candidates merged into its query, or never primed. */
    finished_k,

    /** The key rows of its hinted landmarks are in flight in hop slot zero, each compared to its
     *  hint before the walk seeds. */
    confirming_k,

    /** Its landmarks are in flight in hop slot zero, and every entry that arrives enters the
     *  frontier; spare hop slots may already expand the landmarks that finished. */
    seeding_k,

    /** Its landmarks all arrived, and its expansions, or the scores they issued, are in
     *  flight. */
    expanding_k,

    /** A sated fold or a refused re-table stopped it; its outstanding reads finish unscored, then
     *  it finishes. */
    draining_k,
};

/** Why a walk's read was issued, which routes the fetched row. */
enum class nvme_purpose_t : u8_t {

    /** A value row, scored against the query. */
    score_k,

    /** A neighbors row, extending the frontier. */
    expand_k,

    /** A key row, for a keyed fold or a written result. */
    key_k,

    /** A hinted landmark's key row, compared to its hint before its walk seeds. */
    confirm_k,
};

/** One scored, expanded or resolved entry in flight. Its first read arrives in the buffer of the
 *  flight's own index, and a keyed score's key in the buffer one flight count past it. The
 *  shard it reads follows from its issuer, which keeps a flight of 32-bit IDs at 16 bytes. */
template <typename node_id_type_>
struct nvme_flight {

    /** The walk that issued a score or an expansion, or the query slot behind a resolution. */
    u32_t issuer = 0;

    /** The entry's node ID inside its shard. */
    node_id_type_ node = 0;

    /** Where in its issuer the fetched row counts: a walk's hop slot, or a query slot's output. */
    u32_t position = 0;

    /** Reads still to arrive: two for a keyed score, one otherwise. */
    u8_t reads_left = 0;

    /** What its fetched row does: @c score_k scores, @c expand_k expands, @c key_k resolves,
     *  @c confirm_k confirms a hinted landmark. */
    nvme_purpose_t purpose = nvme_purpose_t::score_k;
};

static_assert(sizeof(nvme_flight<u32_t>) == 16, "A flight is 16 bytes, so none straddles a cache line");

/** A node one walk scored, which an aggregating predicate carries beside its key until the walker
 *  settles the batch it finished in; flat, so a 32-bit node and a 64-bit distance fit 16 bytes. */
template <typename node_id_type_, typename distance_type_>
struct nvme_walk_payload {

    /** The walk that scored it, positional among its thread's walks. */
    u32_t walk = 0;

    /** The node scored. */
    node_id_type_ node_id {};

    /** Its distance. */
    distance_type_ distance {};
};

/** Where a query slot stands. */
enum class nvme_query_stage_t : u8_t {

    /** Free for the next query. */
    idle_k,

    /** Some of its walks are still going. */
    walking_k,

    /** Every walk finished, and its written locators are resolving into keys. */
    resolving_k,
};

/** One query in flight on a thread: its walks still going, and its resolution's progress. */
struct nvme_query_slot_t {

    /** Walks of the query not finished yet. */
    u32_t walks_left = 0;

    /** Locators the query asks to resolve. */
    u32_t outputs = 0;

    /** The next locator to issue a key read for. */
    u32_t next_output = 0;

    /** Key reads of the query in flight. */
    u32_t resolving = 0;

    /** Where the slot stands. */
    nvme_query_stage_t stage = nvme_query_stage_t::idle_k;

    /** The batch query the slot holds. */
    std::size_t query_index = 0;

    /** When the query's walks primed, which its latency counts from; unset when the call collects
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

/** The reads each engine of @p executor keeps in flight for a call over @p media holding @p parts:
 *  the depth the executor pins, else what the media's block queues admit per thread, else, for
 *  media with no block queue, the first hop over the most neighbors any node of @p parts holds. */
template <typename executor_type_, typename part_type_>
std::size_t nvme_call_depth(executor_type_ const& executor, std::span<typename executor_type_::medium_t const> media,
                            std::span<part_type_ const> parts) noexcept {
    if (std::size_t const pinned = executor.pinned_depth()) return pinned;
    if (std::size_t const admitted = nvme_queue_depth(media, executor.size())) return admitted;
    return nvme_first_hop(nvme_widest_row(parts));
}

/**
 *  @brief Claims @p executor, fits its engines @p queue_depth reads deep under the reaping @p media
 *      admit, sizes their arenas and staging buffers, and attaches @p media.
 *
 *  Every thread's arena holds @p scratch_bytes and every engine @p buffers staging buffers of
 *  @p buffer_bytes. Nothing rewinds the arenas again before the call ends, so whatever a call
 *  carves lasts all of it.
 *
 *  @par Statuses:
 *  - @c contended_k Another call holds @p executor, or its pool is running another dispatch.
 *  - @c uninitialized_k The executor holds no threads, or its pool has no workers.
 *  - @c io_k An engine's setup was refused at the fitted shape.
 *  - @c overflow_k A staging region's byte count overflows, or passes what Windows registers.
 *  - @c config_mismatch_k A medium the engines cannot read.
 */
template <typename executor_type_>
expected<nvme_call<executor_type_>> nvme_begin_call(executor_type_& executor,
                                                    std::span<typename executor_type_::medium_t const> media,
                                                    std::size_t queue_depth, std::size_t scratch_bytes,
                                                    std::size_t buffers, std::size_t buffer_bytes) noexcept {
    nvme_call<executor_type_> call;
    auto [claim, claim_status] = executor.claim();
    if (failed(claim_status)) return {std::move(call), claim_status};
    call.claim = std::move(claim);
    if (status_t const fitted = executor.fit(queue_depth, nvme_reaping(media)); failed(fitted))
        return {std::move(call), fitted};
    if (status_t const reserved = executor.reserve(scratch_bytes, buffers, buffer_bytes); failed(reserved))
        return {std::move(call), reserved};
    auto [attachment, attach_status] = executor.attach(media);
    call.attachment = std::move(attachment);
    return {std::move(call), attach_status};
}

/**
 *  @brief One thread's share of a call: its walks, its queries, its flights and the reads not yet
 *      enqueued, driven until the thread's queries run out.
 *
 *  @tparam tier_type_ The index whose vocabulary the walk speaks and whose @c score_staged_ scores
 *      its fetched value rows; it befriends the walker, as the walker befriends it.
 *  @tparam query_type_ One query of the batch: its prepared query, its fold, its bound and its
 *      outputs.
 *  @tparam cache_hints_type_ The line hints streaming a fetched row toward the core while its quad
 *      fills.
 */
template <typename tier_type_, typename query_type_, typename cache_hints_type_ = dummy_cache_hints_t>
class query_walker {
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
    using query_t = query_type_;
    using key_t = typename query_t::key_t;
    using inputs_t = typename query_t::inputs_t;

    /** The walker of executor thread @p thread, whose queries are built from @p inputs, and which
     *  asks that thread's predicate among them. */
    query_walker(engine_t& engine, arena_t& arena, std::span<medium_t const> media, std::span<part_t const> parts,
                 walk_shape_t const& shape, inputs_t const& inputs, std::size_t thread, std::size_t line_bytes) noexcept
        : engine_(engine), arena_(arena), media_(media), parts_(parts), shape_(shape), inputs_(inputs), thread_(thread),
          prefetcher_ {line_bytes} {}

    query_walker(query_walker const&) = delete;
    query_walker& operator=(query_walker const&) = delete;

    ~query_walker() noexcept {
        for (walk_t& walk : walks_) walk.~walk_t();
        for (query_t& query : queries_) query.~query_t();
    }

    /** Carves every walk, query and flight of the thread, and reserves an aggregating predicate for
     *  the widest batch it can gather: every flight of the thread arriving at once. */
    status_t reserve() noexcept {
        std::size_t const parts = parts_.size(), queries = shape_.queries_per_thread;
        std::size_t const walks = queries * parts, flights = walks * shape_.walk_flights;
        walk_t* const walk_slots = carve_<walk_t>(walks);
        query_t* const query_slots = carve_<query_t>(queries);
        nvme_query_slot_t* const slot_states = carve_<nvme_query_slot_t>(queries);
        flight_t* const flight_states = carve_<flight_t>(flights);
        u32_t* const free_flights = carve_<u32_t>(flights);
        u32_t* const hop_flights = carve_<u32_t>(walks * shape_.hop_slots);
        io_request_t* const backlog = carve_<io_request_t>(shape_.buffers);
        io_completion_t* const completions = carve_<io_completion_t>(shape_.buffers);
        if (!walk_slots || !query_slots || !slot_states || !flight_states || !free_flights || !hop_flights ||
            !backlog || !completions)
            return status_t::bad_alloc_k;

        scratch_allocator_t const allocator {arena_};
        nvme_queue_sizes_t const queue_sizes = shape_.queue_sizes;
        std::size_t const landmarks = nvme_landmarks(nvme_widest_row(parts_));
        for (std::size_t walk = 0; walk != walks; ++walk) {
            walk_t& constructed = *new (walk_slots + walk) walk_t {allocator};
            walks_ = {walk_slots, walk + 1};
            if (failed(constructed.results.reserve(queue_sizes.results)) ||
                failed(constructed.frontier.reserve(queue_sizes.frontier)) ||
                failed(constructed.visited.reserve(queue_sizes.visited)) ||
                (query_t::admitting_k && failed(constructed.admitted.reserve(queue_sizes.results))))
                return status_t::bad_alloc_k;
            if constexpr (requires { constructed.seed_slots; }) {
                nvme_seed<node_id_t>* const seed_slots = carve_<nvme_seed<node_id_t>>(landmarks);
                if (!seed_slots) return status_t::bad_alloc_k;
                constructed.seed_slots = {seed_slots, landmarks};
            }
        }
        for (std::size_t query = 0; query != queries; ++query) {
            query_t& constructed = *new (query_slots + query) query_t {inputs_, allocator};
            queries_ = {query_slots, query + 1};
            if (status_t const reserved = constructed.reserve(parts, nvme_widest_row(parts_)); failed(reserved))
                return reserved;
        }
        for (std::size_t query = 0; query != queries; ++query) new (slot_states + query) nvme_query_slot_t {};
        for (std::size_t flight = 0; flight != flights; ++flight)
            free_flights[flight] = static_cast<u32_t>(flights - 1 - flight);
        slots_ = {slot_states, queries};
        flights_ = {flight_states, flights};
        free_flights_ = {free_flights, flights};
        hop_flights_ = {hop_flights, walks * shape_.hop_slots};
        free_count_ = flights;
        backlog_ = {backlog, shape_.buffers};
        completions_ = {completions, shape_.buffers};
        if constexpr (query_t::admitting_k)
            return reserve_predicate(thread_predicate(*inputs_.predicate, thread_), flights);
        return status_t::success_k;
    }

    /** Primes queries until the batch runs out, keeping every slot's walks in flight, and drains
     *  the engine before it returns, whatever happened; a failure on any thread stops this one too.
     *  Answers the first query that refused, as a failure is published to every thread instead. */
    status_t drive_walks() noexcept {
        while (true) {
            if (succeeded(failure_)) {
                if (status_t const shared = shared_failure_(); failed(shared)) failure_ = shared;
                else prime_queries_();
            }
            if (failed(failure_) && !outstanding_) break;
            if (succeeded(failure_)) {
                if (!queries_in_flight_ && exhausted_()) break;
                enqueue_backlog_();
            }
            if (!outstanding_) {
                assert((failed(failure_) || !queries_in_flight_) && "A query in flight always waits on a read");
                continue;
            }
            if (status_t const submitted = engine_.submit(io_wait_t::blocking_k); failed(submitted)) {
                // A refused submission returns with nothing in flight and nothing left to poll.
                fail(submitted);
                outstanding_ = 0;
                break;
            }
            telemetry_.round_trips += queries_in_flight_;
            for (std::size_t polled = engine_.poll_completions(completions_); polled;
                 polled = engine_.poll_completions(completions_)) {
                // Completions of walk reads move to the front, as their flights may serve
                // other queries by the end of the batch.
                std::size_t advancing = 0;
                for (io_completion_t const& completion : completions_.first(polled))
                    if (route_completion_(completion)) completions_[advancing++] = completion;
                // Walks advance once the whole batch finished and its admissions settled, so their
                // next hops weigh every query that did.
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
     *  queries. */
    void fail(status_t failure) noexcept {
        if (succeeded(failure_)) failure_ = failure;
        keep_first_failure<atomic_ref>(*shape_.failure, failure);
    }

    /**
     *  @brief Walks every query of a batch over every part in @p parts, the queries' walks
     *      interleaved on every thread of @p executor, each query built from @p inputs as
     *      @p query_type_ says.
     *
     *  Runs inside a call @ref nvme_begin_call began at @p shape 's sizes, and sums the threads'
     *  telemetry; @p counts are the queries' own, summed into the result's @c count. @p latencies
     *  takes one duration per query, or is empty. Every thread streams its fetched entries through
     *  @p prefetcher.
     */
    template <typename executor_type_>
    static expected<search_result_t> run_walks(std::span<medium_t const> media, std::span<part_t const> parts,
                                               executor_type_& executor, walk_shape_t shape,
                                               typename query_t::inputs_t const& inputs,
                                               std::span<std::size_t const> counts, std::span<u64_t> latencies,
                                               std::size_t line_bytes = default_alignment_k) noexcept {
        search_result_t merged;
        if (!latencies.empty() && latencies.size() != shape.queries_count) return {merged, status_t::config_mismatch_k};
        alignas(default_alignment_k) std::size_t cursor = 0;
        status_t failure = status_t::success_k;
        shape.cursor = &cursor;
        shape.failure = &failure;
        shape.latencies = latencies;
        status_t refusal = status_t::success_k;
        auto add = [](std::size_t& into, std::size_t value) noexcept {
            atomic_ref<std::size_t>(into).store_add(value, std::memory_order_relaxed);
        };
        auto add_reads = [&](nvme_reads_t& into, nvme_reads_t const& reads) noexcept {
            add(into.count, reads.count), add(into.bytes, reads.bytes);
        };
        auto walk_thread = [&](std::size_t thread, std::size_t) noexcept {
            query_walker walker {
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
            keep_first_failure<atomic_ref>(refusal, refused);
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
        return atomic_ref<std::size_t>(*shape_.cursor).load(std::memory_order_relaxed) >= shape_.queries_count;
    }

    /** Hop slots a walk may hold now: its own share, and once the batch ran out, the shares of the
     *  query slots that went idle; the shares only grow after that, so the free list never runs
     *  dry. */
    std::size_t live_hops_() const noexcept {
        std::size_t const queries = shape_.queries_per_thread;
        if (queries_in_flight_ == queries || !exhausted_()) return shape_.hops_per_walk;
        std::size_t const inherited = (queries - queries_in_flight_) * shape_.walk_flights /
                                      (queries_in_flight_ * (nvme_widest_row(parts_) + 1));
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

    /** Stops @p walk after a scratch refusal, counted as saturated: its outstanding reads arrive
     *  unscored, then it finishes. */
    void saturate_(walk_t& walk) noexcept {
        ++telemetry_.saturated_walks;
        walk.stage = nvme_walk_stage_t::draining_k;
    }

    /** The buffer a keyed score's key arrives in: one flight count past its value's. */
    io_buffer_id_t key_buffer_(u32_t flight_index) const noexcept {
        return static_cast<io_buffer_id_t>(flight_index + flights_.size());
    }

    /** The shard @p flight reads: its walk's, or the shard of the output a resolution fills. */
    u32_t shard_of_(flight_t const& flight) const noexcept {
        return flight.purpose == nvme_purpose_t::key_k ? queries_[flight.issuer].output_locator(flight.position).shard
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

    /** The fetched row of @p flight_index in the column @p purpose read, inside @p buffer. */
    byte_t const* fetched_row_(u32_t flight_index, nvme_purpose_t purpose, io_buffer_id_t buffer) noexcept {
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
        nvme_reads_t& reads = purpose == nvme_purpose_t::expand_k  ? telemetry_.neighbor_reads
                              : purpose == nvme_purpose_t::score_k ? telemetry_.value_reads
                                                                   : telemetry_.key_reads;
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

    /** Takes the next queries of the batch into every idle slot and seeds all of their walks. */
    void prime_queries_() noexcept {
        std::size_t const parts = parts_.size();
        for (u32_t slot_index = 0; slot_index != slots_.size(); ++slot_index) {
            nvme_query_slot_t& slot = slots_[slot_index];
            if (slot.stage != nvme_query_stage_t::idle_k) continue;
            std::size_t const query = atomic_ref<std::size_t>(*shape_.cursor).fetch_add(1, std::memory_order_relaxed);
            if (query >= shape_.queries_count) return;
            slot = nvme_query_slot_t {static_cast<u32_t>(parts), 0, 0, 0, nvme_query_stage_t::walking_k, query, {}};
            if (!shape_.latencies.empty()) slot.primed = std::chrono::steady_clock::now();
            ++queries_in_flight_;
            query_t& primed = queries_[slot_index];
            primed.prime_query(query);
            tier_t::prime_scoring_(shape_, primed);
            for (u32_t shard = 0; shard != parts; ++shard) {
                u32_t const walk_index = slot_index * static_cast<u32_t>(parts) + shard;
                walk_t& walk = walks_[walk_index];
                walk.results.clear();
                walk.admitted.clear();
                walk.frontier.clear();
                walk.visited.clear();
                walk.staged.clear();
                walk.random_generator = seeded_stream(u64_t(query) * parts + shard + 1);
                walk.shard = shard;
                walk.query_slot = slot_index;
                walk.stage = nvme_walk_stage_t::seeding_k;
                std::span<u32_t> const hop_flights = hop_flights_of_(walk_index);
                std::memset(hop_flights.data(), 0, hop_flights.size_bytes());
                if constexpr (requires { walk.seeds; }) walk.seeds = {};
                if constexpr (requires { primed.find_hinted(shard, walk.seed_slots); }) {
                    walk.seeds = walk.seed_slots.first(
                        primed.find_hinted(shard, walk.seed_slots.first(nvme_landmarks(parts_[shard].connectivity))));
                    if (!walk.seeds.empty() && primed.confirms_hints()) {
                        confirm_seeds_(walk_index);
                        continue;
                    }
                }
                if (!seed_landmarks_(walk_index, nvme_landmarks(parts_[shard].connectivity))) finish_walk_(walk_index);
            }
        }
    }

    /** Issues up to @p count fresh landmarks of walk @p walk_index as one hop, answering how
     *  many: its seeds first, each taken once and skipped when already visited, then uniform
     *  draws. A walk seeds only with nothing in flight, so the hop takes slot zero. */
    std::size_t seed_landmarks_(u32_t walk_index, std::size_t count) noexcept {
        walk_t& walk = walks_[walk_index];
        node_id_t const nodes = parts_[walk.shard].nodes;
        if (failed(walk.visited.reserve(walk.visited.size() + count))) return saturate_(walk), 0;
        std::size_t issued = 0;
        while (issued != count && walk.visited.size() < nodes) {
            node_id_t landmark = nodes;
            if constexpr (requires { walk.seeds; })
                if (!walk.seeds.empty()) {
                    landmark = walk.seeds.front().node;
                    walk.seeds = walk.seeds.subspan(1);
                    if (walk.visited.contains(landmark)) continue;
                }
            if (landmark == nodes) {
                landmark = uniform_below<node_id_t> {nodes}(walk.random_generator);
                while (walk.visited.contains(landmark)) landmark = landmark + 1 == nodes ? 0 : landmark + 1;
            }
            [[maybe_unused]] membership_t const membership = walk.visited.insert(landmark);
            issue_score_(walk_index, landmark, 0);
            ++issued;
        }
        std::span<u32_t> const hop_flights = hop_flights_of_(walk_index);
        hop_flights[0] = static_cast<u32_t>(issued);
        return issued;
    }

    /** Issues one key read per seed of walk @p walk_index in hop slot zero, each confirming its
     *  seed's hint before the walk seeds. */
    void confirm_seeds_(u32_t walk_index) noexcept {
        walk_t& walk = walks_[walk_index];
        walk.stage = nvme_walk_stage_t::confirming_k;
        for (u32_t slot = 0; slot != walk.seeds.size(); ++slot) {
            u32_t const flight = take_flight_();
            flights_[flight] = flight_t {walk_index, walk.seeds[slot].node, slot, 1, nvme_purpose_t::confirm_k};
            issue_row_read_(flight, nvme_purpose_t::confirm_k, flight);
        }
        std::span<u32_t> const hop_flights = hop_flights_of_(walk_index);
        hop_flights[0] = static_cast<u32_t>(walk.seeds.size());
    }

    /** Keeps the seed of @p flight_index when its fetched key is its hint's, and once the last of
     *  its walk's seeds arrived, seeds the walk from the kept ones ahead of its draws. */
    void confirm_fetched_key_(u32_t flight_index) noexcept {
        if constexpr (requires(query_t& query, key_t const& key) { query.confirm_hint(u32_t {}, key); }) {
            flight_t const& flight = flights_[flight_index];
            u32_t const walk_index = flight.issuer;
            walk_t& walk = walks_[walk_index];
            nvme_seed<node_id_t>& seed = walk.seed_slots[flight.position];
            key_t const stored = safegraphs_widened_key<key_t>(
                fetched_row_(flight_index, nvme_purpose_t::confirm_k, flight_index), parts_[walk.shard].key_dtype);
            if (!queries_[walk.query_slot].confirm_hint(seed.hint, stored)) seed.node = parts_[walk.shard].nodes;
            release_flight_(flight_index);
            std::span<u32_t> const hop_flights = hop_flights_of_(walk_index);
            if (--hop_flights[0]) return;
            std::size_t kept = 0;
            for (nvme_seed<node_id_t> const candidate : walk.seeds)
                if (candidate.node != parts_[walk.shard].nodes) walk.seed_slots[kept++] = candidate;
            walk.seeds = walk.seed_slots.first(kept);
            walk.stage = nvme_walk_stage_t::seeding_k;
            if (!seed_landmarks_(walk_index, nvme_landmarks(parts_[walk.shard].connectivity))) finish_walk_(walk_index);
        }
    }

    void issue_score_(u32_t walk_index, node_id_t node, u32_t hop) noexcept {
        u32_t const flight = take_flight_();
        flights_[flight] = flight_t {walk_index, node, hop, static_cast<u8_t>(shape_.buffers_per_flight),
                                     nvme_purpose_t::score_k};
        issue_row_read_(flight, nvme_purpose_t::score_k, flight);
        if constexpr (query_t::key_reads_k == nvme_key_reads_t::scored_k)
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
            prefetcher_.prefetch(fetched_row_(flight_index, nvme_purpose_t::score_k, flight_index),
                                 row_bytes_(flight, nvme_purpose_t::score_k));
            walk_t& walk = walks_[flight.issuer];
            std::span<u32_t> const hop_flights = hop_flights_of_(flight.issuer);
            [[maybe_unused]] fu::status_t const queued = walk.staged.push_back(flight_index);
            // A quad scores once full or once a hop's last row arrives, whichever hops its lanes
            // came from.
            if (--hop_flights[flight.position] && !walk.staged.full()) return true;
            score_staged_flights_(walk);
            // Landmarks take hop slot zero, so seeding ends once that slot drains.
            if (walk.stage == nvme_walk_stage_t::seeding_k && !hop_flights[0])
                walk.stage = nvme_walk_stage_t::expanding_k;
            return true;
        }
        case nvme_purpose_t::expand_k: expand_fetched_row_(flight.issuer, flight_index); return true;
        case nvme_purpose_t::key_k: {
            u32_t const slot_index = flight.issuer;
            byte_t const* const stored = fetched_row_(flight_index, nvme_purpose_t::key_k, flight_index);
            dtype_t const stored_dtype = parts_[shard_of_(flight)].key_dtype;
            queries_[slot_index].resolve_output_key(flight.position,
                                                    safegraphs_widened_key<key_t>(stored, stored_dtype));
            release_flight_(flight_index);
            --slots_[slot_index].resolving;
            resolve_query_(slot_index);
            return false;
        }
        case nvme_purpose_t::confirm_k: confirm_fetched_key_(flight_index); return false;
        }
        return false;
    }

    /** Releases the staged flights of a draining @p walk unscored, or hands them to the tier, which
     *  scores them as one quad, folds them through the walk's query, releases them and ranks the
     *  rest through @c rank_. */
    void score_staged_flights_(walk_t& walk) noexcept {
        if (!walk.staged.size()) return;
        if (walk.stage == nvme_walk_stage_t::draining_k) {
            for (u32_t const flight : walk.staged) release_flight_(flight);
            return walk.staged.clear();
        }
        tier_t::score_staged_(*this, walk);
    }

    /** Merges the ascending @p scored entries of @p walk into its candidates, and queues them for
     *  expansion: every landmark, as the in-memory walk seeds the frontier, and a neighbor only
     *  when the candidates took it. */
    [[gnu::always_inline]] void rank_(walk_t& walk, std::span<scored_node_t const> scored) noexcept {
        std::size_t const accepted = walk.results.insert(scored, shape_.expansion_factor);
        std::size_t const queued = walk.stage == nvme_walk_stage_t::seeding_k ? scored.size() : accepted;
        for (std::size_t lane = 0; lane != queued; ++lane)
            if (failed(walk.frontier.insert(scored_node_t(scored[lane])))) return saturate_(walk);
    }

    /** Admits the nodes of @p scored, scored from the staged flights of @p walk in lane order, that
     *  could still enter its admitted set and that the query's relation and the thread's predicate
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

    /** Asks the query's relation about the fetched entries of the contending @p flights, then the
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

    /** Without a predicate every related node is admitted, and no key was fetched to ask about. */
    template <typename predicate_type_, std::size_t extent_>
        requires std::same_as<std::remove_cv_t<predicate_type_>, dummy_predicate_t>
    void ask_predicate_(predicate_type_&, std::span<u32_t const, extent_>,
                        std::span<walk_payload_t const, extent_> payloads) noexcept {
        for (walk_payload_t const& payload : payloads) admit_payload_(payload);
    }

    /** Asks @p predicate about the keys fetched beside @p flights, each carrying its payload. */
    template <typename predicate_type_, std::size_t extent_>
    void ask_predicate_(predicate_type_& predicate, std::span<u32_t const, extent_> flights,
                        std::span<walk_payload_t const, extent_> payloads) noexcept {
        std::array<key_t, extent_> keys;
        for (std::size_t lane = 0; lane != extent_; ++lane) keys[lane] = fetched_key_(flights[lane]);
        ask_quad(predicate, std::span<key_t const, extent_>(keys), payloads,
                 [this](walk_payload_t const& payload) noexcept { admit_payload_(payload); });
    }

    /** Answers what the thread's predicate gathered over a completion batch, so every walk advances
     *  on its whole admitted set. */
    void settle_admissions_() noexcept {
        if constexpr (query_t::admitting_k)
            settle_predicate(thread_predicate(*inputs_.predicate, thread_),
                             [this](walk_payload_t const& payload) noexcept { admit_payload_(payload); });
    }

    /** Inserts an admitted node into the admitted set of the walk that scored it. */
    void admit_payload_(walk_payload_t const& payload) noexcept {
        [[maybe_unused]] bool const inserted = walks_[payload.walk].admitted.insert(
            scored_node_t {payload.node_id, payload.distance}, shape_.expansion_factor);
    }

    /** The key fetched beside the value of the keyed score @p flight_index. */
    key_t fetched_key_(u32_t flight_index) noexcept {
        return safegraphs_widened_key<key_t>(
            fetched_row_(flight_index, nvme_purpose_t::key_k, key_buffer_(flight_index)),
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
        query_t const& query = queries_[walk.query_slot];
        std::size_t const expansion_factor = shape_.expansion_factor;
        results_array_t const& stopping = query.stopping_candidates(walk);
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

    /** Reads the fetched neighbors row of @p flight_index and issues a score for every unvisited
     *  neighbor, in the expansion's own hop. */
    void expand_fetched_row_(u32_t walk_index, u32_t flight_index) noexcept {
        walk_t& walk = walks_[walk_index];
        part_t const& part = parts_[walk.shard];
        u32_t const hop = flights_[flight_index].position;
        std::span<u32_t> const hop_flights = hop_flights_of_(walk_index);
        --hop_flights[hop];
        if (walk.stage != nvme_walk_stage_t::draining_k) {
            if (failed(walk.visited.reserve(walk.visited.size() + part.connectivity))) saturate_(walk);
            else {
                auto const* const neighbors = reinterpret_cast<node_id_t const*>(
                    fetched_row_(flight_index, nvme_purpose_t::expand_k, flight_index));
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

    /** Merges @p walk_index into its query, and once every walk of the query finished, starts the
     *  query's resolution. */
    void finish_walk_(u32_t walk_index) noexcept {
        walk_t& walk = walks_[walk_index];
        walk.stage = nvme_walk_stage_t::finished_k;
        u32_t const slot_index = walk.query_slot;
        nvme_query_slot_t& slot = slots_[slot_index];
        assert(slot.stage == nvme_query_stage_t::walking_k && "Only a walking query's walks finish");
        queries_[slot_index].finish_walk(walk);
        if (--slot.walks_left) return;
        slot.stage = nvme_query_stage_t::resolving_k;
        slot.outputs = static_cast<u32_t>(queries_[slot_index].output_count());
        resolve_query_(slot_index);
    }

    /** Issues key reads for the query's written locators within its flight budget, and
     *  finishes the query once every one finished. */
    void resolve_query_(u32_t slot_index) noexcept {
        nvme_query_slot_t& slot = slots_[slot_index];
        assert(slot.stage == nvme_query_stage_t::resolving_k && "Only a resolving query reads keys");
        query_t& query = queries_[slot_index];
        std::size_t const budget = parts_.size() * shape_.walk_flights;
        while (slot.next_output != slot.outputs && slot.resolving != budget) {
            auto const locator = query.output_locator(slot.next_output);
            u32_t const flight = take_flight_();
            flights_[flight] = flight_t {slot_index, locator.node, slot.next_output, 1, nvme_purpose_t::key_k};
            issue_row_read_(flight, nvme_purpose_t::key_k, flight);
            ++slot.next_output;
            ++slot.resolving;
        }
        if (slot.resolving) return;
        if (status_t const finished = query.finish_query(); failed(finished) && succeeded(refusal_))
            refusal_ = finished;
        if (!shape_.latencies.empty())
            shape_.latencies[slot.query_index] = static_cast<u64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - slot.primed)
                    .count());
        slot.stage = nvme_query_stage_t::idle_k;
        --queries_in_flight_;
    }

    engine_t& engine_;
    arena_t& arena_;
    std::span<medium_t const> media_;
    std::span<part_t const> parts_;
    walk_shape_t const& shape_;
    inputs_t const& inputs_;

    /** The executor thread this walker runs on, whose predicate it asks. */
    std::size_t thread_ = 0;

    /** Streams each finished row toward the core while the rest of its quad is still in flight. */
    object_prefetcher<cache_hints_type_> prefetcher_;

    std::span<walk_t> walks_;
    std::span<query_t> queries_;
    std::span<nvme_query_slot_t> slots_;
    std::span<flight_t> flights_;
    std::span<u32_t> free_flights_;
    std::span<u32_t> hop_flights_;
    std::span<io_request_t> backlog_;
    std::span<io_completion_t> completions_;
    std::size_t free_count_ = 0;
    std::size_t backlog_size_ = 0;
    std::size_t outstanding_ = 0;
    std::size_t queries_in_flight_ = 0;
    search_result_t telemetry_;

    /** The read failure that stops this thread. */
    status_t failure_ = status_t::success_k;

    /** The first query that refused, which leaves the other queries standing. */
    status_t refusal_ = status_t::success_k;
};

#pragma endregion Walks

#pragma region Scans

/** One staging slot of a scan: the task whose reads it holds, the task's progress as its describe
 *  found it, what each read asked for, where each one's bytes start in its covering read, and the
 *  reads still to arrive. */
template <std::size_t reads_per_task_>
struct nvme_scan_slot {
    std::size_t task = 0;
    u64_t progress = 0;
    std::array<nvme_extent_t, reads_per_task_> extents {};
    std::array<u64_t, reads_per_task_> offsets {};
    std::size_t reads_left = 0;
};

/** Arena bytes one @c nvme_scan_reads carves on every thread of an engine @p depth reads deep. */
template <std::size_t reads_per_task_>
constexpr std::size_t nvme_scan_bytes(std::size_t depth) noexcept {
    using slice_t = arena_allocator<byte_t>;
    return slice_t::reserved_bytes(depth * sizeof(nvme_scan_slot<reads_per_task_>)) +
           slice_t::reserved_bytes(depth * sizeof(u32_t)) +
           slice_t::reserved_bytes(depth * reads_per_task_ * sizeof(io_completion_t));
}

/** Begins a call of @p scans scans of @p reads_per_task_ reads per task over @p media, as
 *  @ref nvme_begin_call begins one: every engine @p depth slots deep, each slot as many staging
 *  buffers of @p buffer_bytes as reads, at least @p buffers for a walk sharing the call, and every
 *  arena sized for the slots and @p scratch_bytes more. */
template <std::size_t reads_per_task_ = 1, typename executor_type_>
expected<nvme_call<executor_type_>> nvme_begin_scans(executor_type_& executor,
                                                     std::span<typename executor_type_::medium_t const> media,
                                                     std::size_t depth, std::size_t scans, std::size_t scratch_bytes,
                                                     std::size_t buffers, std::size_t buffer_bytes) noexcept {
    return nvme_begin_call(executor, media, depth,
                           scans * nvme_scan_bytes<reads_per_task_>(depth) + scratch_bytes + default_alignment_k,
                           max_of(depth * reads_per_task_, buffers), buffer_bytes);
}

/**
 *  @brief One thread's share of @c nvme_scan_reads: its engine and slots, the task it claimed and
 *      describes, the slot whose reads it enqueues, and the reads it issued.
 *
 *  A thread claims a task whole and describes it slot by slot until an empty first extent ends it.
 *  Read r of slot s arrives in buffer s + r · depth, so the buffer alone names what arrived, and a
 *  slot's reads are consumed together once the last one lands.
 */
template <std::size_t reads_per_task_, template <typename> class atomic_ref_template_, typename engine_type_,
          typename describe_type_, typename consume_type_>
struct nvme_scan_lane {
    using slot_t = nvme_scan_slot<reads_per_task_>;
    using medium_t = typename engine_type_::medium_t;
    template <typename word_type_>
    using atomic_ref = atomic_ref_template_<word_type_>;

    /** The task a lane holds when it holds none. */
    inline static constexpr std::size_t unclaimed_k = ~std::size_t {0};

    engine_type_& engine;
    std::span<medium_t const> media;
    describe_type_& describe;
    consume_type_& consume;

    /** The next task any thread claims, and the first failure any thread kept. */
    std::size_t& cursor;
    status_t const& failure;

    std::size_t thread = 0;
    std::size_t depth = 0;
    std::size_t tasks_count = 0;

    /** The staging slots, the free ones' indices, and room for one poll of every read in flight. */
    slot_t* slots = nullptr;
    u32_t* free_slots = nullptr;
    io_completion_t* completions = nullptr;

    status_t local = status_t::success_k;
    std::size_t free_count = 0, outstanding = 0;
    std::size_t task = unclaimed_k;
    u64_t progress = 0;

    /** The slot whose reads are being enqueued and its next read, which a full engine may refuse
     *  until a completion frees room in it; @c depth while none is. */
    std::size_t issuing = depth, next_read = 0;
    nvme_reads_t reads;

    /** Issues and polls until none of this thread's reads is in flight, every slot free at
     *  first. */
    void run() noexcept {
        for (; free_count != depth; ++free_count) free_slots[free_count] = static_cast<u32_t>(free_count);
        std::span<io_completion_t> const polling {completions, depth * reads_per_task_};
        for (issue(); outstanding; issue()) {
            if (status_t const submitted = engine.submit(io_wait_t::blocking_k); failed(submitted)) {
                // A refused submission returns with nothing in flight and nothing left to poll.
                if (succeeded(local)) local = submitted;
                break;
            }
            for (std::size_t polled = engine.poll_completions(polling); polled;
                 polled = engine.poll_completions(polling))
                for (io_completion_t const& completion : polling.first(polled)) arrive(completion);
        }
    }

    /** Enqueues reads until the engine is full, every slot is busy, the tasks ran out, or a thread
     *  failed. */
    void issue() noexcept {
        while (succeeded(local) && succeeded(atomic_ref<status_t const>(failure).load(std::memory_order_relaxed))) {
            if (issuing == depth) {
                if (!free_count || !claim()) return;
                if (!describe_slot()) continue;
            }
            if (!enqueue()) return;
        }
    }

    /** Keeps the claimed task, or claims the next one at progress zero; false once the tasks ran
     *  out. */
    bool claim() noexcept {
        if (task == unclaimed_k)
            task = min_of(atomic_ref<std::size_t>(cursor).fetch_add(1, std::memory_order_relaxed), tasks_count),
            progress = 0;
        return task != tasks_count;
    }

    /** Asks for the claimed task's next reads into a free slot, @c progress as describe found it
     *  kept for consume; false, releasing the task, once it ended. */
    bool describe_slot() noexcept {
        u64_t const described = progress;
        std::array<nvme_extent_t, reads_per_task_> const extents = describe(task, progress);
        if (!extents[0].length) return task = unclaimed_k, false;
        issuing = free_slots[--free_count];
        next_read = 0;
        slots[issuing] = slot_t {task, described, extents, {}, reads_per_task_};
        return true;
    }

    /** Enqueues the issuing slot's next read; false when the engine refused it. */
    bool enqueue() noexcept {
        slot_t& slot = slots[issuing];
        nvme_extent_t const& extent = slot.extents[next_read];
        medium_t const& medium = media[extent.medium];
        u64_t const begin = byte_base_of(medium) + extent.begin;
        // The buffer alone names what arrived, so the context stays unused.
        io_request_t const request = nvme_covering_read(begin, extent.length, block_bytes_of(medium), extent.medium,
                                                        static_cast<io_buffer_id_t>(issuing + next_read * depth), 0);
        slot.offsets[next_read] = begin - request.offset;
        status_t const enqueued = engine.enqueue_read(request);
        if (enqueued == status_t::overflow_k && outstanding) return false;
        // A full engine holding none of this thread's reads would never drain, so it fails instead.
        if (failed(enqueued)) return local = enqueued == status_t::overflow_k ? status_t::io_k : enqueued, false;
        reads += nvme_reads_t {1, request.length};
        ++outstanding;
        if (++next_read == reads_per_task_) issuing = depth;
        return true;
    }

    /** Takes one completion, and once its slot's last read arrived, consumes the slot and frees
     *  it. */
    void arrive(io_completion_t const& completion) noexcept {
        --outstanding;
        std::size_t const slot_index = completion.buffer % depth, read = completion.buffer / depth;
        slot_t& slot = slots[slot_index];
        u64_t const needed = slot.offsets[read] + slot.extents[read].length;
        if (succeeded(local) && (completion.result < 0 || static_cast<u64_t>(completion.result) < needed))
            local = status_t::io_k;
        if (--slot.reads_left) return;
        if (succeeded(local)) {
            std::array<byte_t const*, reads_per_task_> bytes;
            for (std::size_t each = 0; each != reads_per_task_; ++each)
                bytes[each] = engine.buffer(static_cast<io_buffer_id_t>(slot_index + each * depth)) +
                              slot.offsets[each];
            consume(thread, slot.task, slot.progress, slot.extents, bytes);
        }
        free_slots[free_count++] = static_cast<u32_t>(slot_index);
    }
};

/**
 *  @brief Scans the reads of @p tasks_count tasks over every thread of @p executor, as many slots
 *      in flight per thread as its engine's depth and its staging buffers allow, answering how
 *      many reads it issued.
 *
 *  A thread claims a task whole and asks @p describe for its reads in turn, as
 *  <tt>describe(task, progress)</tt> with @p progress zero at the claim and kept between the calls; each
 *  call answers @p reads_per_task_ extents on @p media, every one non-empty, or an empty first one,
 *  which ends the task. The reads of one call share a slot, read r of slot s arriving in buffer s +
 *  r · depth, and @p consume takes them once all arrived, as
 *  <tt>consume(thread, task, progress, extents, bytes)</tt> on the thread that read them, @c progress as
 *  the describe that answered them found it, both arrays in read order. Runs inside a call
 *  @c nvme_begin_scans began for as many reads per task; a failed read stops every thread, which
 *  drains its engine.
 */
template <std::size_t reads_per_task_, template <typename> class atomic_ref_template_, typename executor_type_,
          typename describe_type_, typename consume_type_>
expected<nvme_reads_t> nvme_scan_reads(executor_type_& executor,
                                       std::span<typename executor_type_::medium_t const> media,
                                       std::size_t tasks_count, describe_type_&& describe,
                                       consume_type_&& consume) noexcept {
    using lane_t = nvme_scan_lane<reads_per_task_, atomic_ref_template_, typename executor_type_::engine_t,
                                  describe_type_, consume_type_>;
    using slot_t = typename lane_t::slot_t;
    std::size_t const depth = max_of<std::size_t>(
        min_of(executor.queue_depth(), executor.buffers_cut().buffers / reads_per_task_), 1);
    alignas(default_alignment_k) std::size_t cursor = 0;
    status_t failure = status_t::success_k;
    nvme_reads_t issued;
    auto scan_thread = [&](std::size_t thread, std::size_t) noexcept {
        auto& arena = executor.arena(thread);
        lane_t lane {executor.engine(thread), media, describe, consume, cursor, failure, thread, depth, tasks_count};
        lane.slots = reinterpret_cast<slot_t*>(arena.allocate(depth * sizeof(slot_t), default_alignment_k));
        lane.free_slots = reinterpret_cast<u32_t*>(arena.allocate(depth * sizeof(u32_t), default_alignment_k));
        lane.completions = reinterpret_cast<io_completion_t*>(
            arena.allocate(depth * reads_per_task_ * sizeof(io_completion_t), default_alignment_k));
        if (lane.slots && lane.free_slots && lane.completions) lane.run();
        else lane.local = status_t::bad_alloc_k;
        atomic_ref_template_<std::size_t>(issued.count).store_add(lane.reads.count, std::memory_order_relaxed);
        atomic_ref_template_<std::size_t>(issued.bytes).store_add(lane.reads.bytes, std::memory_order_relaxed);
        keep_first_failure<atomic_ref_template_>(failure, lane.local);
    };
    if (status_t const fanned = executor.for_threads(scan_thread); failed(fanned)) return {issued, fanned};
    return {issued, failure};
}

/** Bytes a read may span on blocks of @p block_bytes and still fit a staging buffer of
 *  @p buffer_bytes, wherever it starts. */
constexpr u64_t nvme_covered_bytes(u64_t buffer_bytes, u64_t block_bytes) noexcept {
    return buffer_bytes / block_bytes * block_bytes - (block_bytes - 1);
}

/** Whether @p extent joins @p read on @p medium: the same medium, blocks that touch the read's, so
 *  the merged read covers no block that a read per extent would not, and still within what a
 *  staging buffer of @p buffer_bytes covers. */
template <typename medium_type_>
bool nvme_extends_read(medium_type_ const& medium, nvme_extent_t const& read, nvme_extent_t const& extent,
                       std::size_t buffer_bytes) noexcept {
    u64_t const block = block_bytes_of(medium), base = byte_base_of(medium);
    return extent.medium == read.medium &&
           (base + extent.begin) / block <= divide_round_up(base + read.begin + read.length, block) &&
           extent.begin + extent.length - read.begin <= nvme_covered_bytes(buffer_bytes, block);
}

/**
 *  @brief Reads the bytes @p extent_of locates for every entry of @p nodes, as
 *      <tt>extent_of(position) noexcept → nvme_extent_t</tt>, and hands each non-empty extent's to
 *      @p on_entry as <tt>on_entry(thread, position, bytes)</tt> on the thread that read them.
 *
 *  The entries sort by locator in thread zero's arena, and extents whose blocks touch merge into
 *  one read while it fits a staging buffer of @p buffer_bytes, the reads taken as the tasks of
 *  @c nvme_scan_reads, inside a call @c nvme_begin_scans began.
 *
 *  @par Statuses:
 *  - @c bad_alloc_k Thread zero's arena could not hold the order.
 *  - @c overflow_k A read finds its engine's queue full.
 *  - @c io_k A read came back short.
 */
template <template <typename> class atomic_ref_template_, typename executor_type_, typename node_id_type_,
          typename extent_of_type_, typename on_entry_type_>
status_t nvme_read_located(executor_type_& executor, std::span<typename executor_type_::medium_t const> media,
                           std::size_t buffer_bytes, std::span<shard_node<node_id_type_> const> nodes,
                           extent_of_type_&& extent_of, on_entry_type_&& on_entry) noexcept {
    using scratch_allocator_t = arena_allocator<byte_t, forkunion_allocator_t>;
    scratch_allocator_t const scratch {executor.arena(0)};
    growing_buffer<std::size_t, scratch_allocator_t> sorted {scratch}, run_starts {scratch};
    if (failed(sorted.resize_uninitialized(nodes.size())) || failed(run_starts.reserve_exactly(nodes.size() + 1)))
        return status_t::bad_alloc_k;
    for (std::size_t position = 0; position != nodes.size(); ++position) sorted[position] = position;
    partial_sort(to_writable_span(sorted), sorted.size(),
                 [&](std::size_t first, std::size_t second) noexcept { return nodes[first] < nodes[second]; });
    nvme_extent_t run;
    for (std::size_t rank = 0; rank != sorted.size(); ++rank) {
        nvme_extent_t const located = extent_of(sorted[rank]);
        if (rank && nvme_extends_read(media[located.medium], run, located, buffer_bytes))
            run.length = static_cast<u32_t>(located.begin + located.length - run.begin);
        else run_starts.push_back_reserved(rank), run = located;
    }
    run_starts.push_back_reserved(sorted.size());
    auto describe = [&](std::size_t run_index, u64_t& progress) noexcept -> std::array<nvme_extent_t, 1> {
        if (progress++) return {};
        nvme_extent_t const first = extent_of(sorted[run_starts[run_index]]);
        nvme_extent_t const last = extent_of(sorted[run_starts[run_index + 1] - 1]);
        return {nvme_extent_t {first.begin, static_cast<u32_t>(last.begin + last.length - first.begin), first.medium}};
    };
    auto consume = [&](std::size_t thread, std::size_t run_index, u64_t, std::array<nvme_extent_t, 1> const& reads,
                       std::array<byte_t const*, 1> const& fetched) noexcept {
        for (std::size_t rank = run_starts[run_index]; rank != run_starts[run_index + 1]; ++rank) {
            nvme_extent_t const located = extent_of(sorted[rank]);
            if (located.length)
                on_entry(thread, sorted[rank],
                         bytes_view_t {fetched[0] + (located.begin - reads[0].begin), located.length});
        }
    };
    return nvme_scan_reads<1, atomic_ref_template_>(executor, media, run_starts.size() - 1, describe, consume).status;
}

/**
 *  @brief The earliest entry of every key of @p keys among @p parts, in part then node order,
 *      through one @p scan_keys of the key columns; a key held nowhere gets the part
 *      @p parts.size().
 *
 *  @p scan_keys runs as <tt>scan_keys(keys, on_hit) noexcept → status_t</tt>, calling
 *  <tt>on_hit(thread, position, locator)</tt> for every entry under every key, from any thread.
 */
template <template <typename> class atomic_ref_template_, typename key_type_, typename node_id_type_,
          typename part_type_, typename executor_type_, typename scan_keys_type_>
status_t nvme_find_first(std::span<key_type_ const> keys, std::span<shard_node<node_id_type_>> locators,
                         std::span<part_type_ const> parts, executor_type_& executor,
                         scan_keys_type_&& scan_keys) noexcept {
    using scratch_allocator_t = arena_allocator<byte_t, forkunion_allocator_t>;
    scratch_allocator_t const scratch {executor.arena(0)};
    growing_buffer<u64_t, scratch_allocator_t> earliest {scratch}, bases {scratch};
    if (failed(earliest.resize_uninitialized(keys.size())) || failed(bases.resize(parts.size() + 1)))
        return status_t::bad_alloc_k;
    // An entry's ordinal orders as its part, then its node, so the lowest ordinal wins.
    for (std::size_t part = 0; part != parts.size(); ++part) bases[part + 1] = bases[part] + parts[part].nodes;
    for (u64_t& ordinal : earliest) ordinal = bases.back();
    status_t const scanned = scan_keys(
        keys, [&](std::size_t, std::size_t position, shard_node<node_id_type_> locator) noexcept {
            atomic_ref_template_<u64_t>(earliest[position])
                .store_min(bases[locator.shard] + locator.node, std::memory_order_relaxed);
        });
    if (failed(scanned)) return scanned;
    for (std::size_t position = 0; position != keys.size(); ++position) {
        u32_t part = 0;
        while (part != parts.size() && earliest[position] >= bases[part + 1]) ++part;
        locators[position] = {part, static_cast<node_id_type_>(earliest[position] - bases[part])};
    }
    return status_t::success_k;
}

/**
 *  @brief Lists every entry of every key of @p keys in one @p scan_keys of the key columns, as
 *      @c nvme_find_first scans them: each key's entries in @p nodes from @p starts, zeroed by
 *      the caller, in part then node order.
 *
 *  Every thread keeps its hits on its own arena, so the scan takes no lock; thread zero then
 *  counts them per key, scatters them and sorts each key's run.
 */
template <typename key_type_, typename node_id_type_, typename nodes_allocator_type_, typename executor_type_,
          typename scan_keys_type_>
status_t nvme_list_entries(std::span<key_type_ const> keys, std::span<std::size_t> starts,
                           growing_buffer<shard_node<node_id_type_>, nodes_allocator_type_>& nodes,
                           executor_type_& executor, scan_keys_type_&& scan_keys) noexcept {
    using shard_node_t = shard_node<node_id_type_>;
    using scratch_allocator_t = arena_allocator<byte_t, forkunion_allocator_t>;
    if (keys.empty()) return status_t::success_k;

    struct listing_share_t {
        growing_buffer<u32_t, scratch_allocator_t> positions;
        growing_buffer<shard_node_t, scratch_allocator_t> nodes;
        status_t failure = status_t::success_k;
    };
    std::size_t const threads = executor.size();
    listing_share_t* const shares =
        arena_allocator<listing_share_t, forkunion_allocator_t> {executor.arena(0)}.allocate(threads);
    if (!shares) return status_t::bad_alloc_k;
    for (std::size_t thread = 0; thread != threads; ++thread)
        new (shares + thread) listing_share_t {
            growing_buffer<u32_t, scratch_allocator_t> {scratch_allocator_t {executor.arena(thread)}},
            growing_buffer<shard_node_t, scratch_allocator_t> {scratch_allocator_t {executor.arena(thread)}}};
    status_t listed = scan_keys(keys, [&](std::size_t thread, std::size_t position, shard_node_t locator) noexcept {
        listing_share_t& share = shares[thread];
        if (failed(share.positions.push_back(static_cast<u32_t>(position))) || failed(share.nodes.push_back(locator)))
            share.failure = status_t::bad_alloc_k;
    });
    for (std::size_t thread = 0; thread != threads && succeeded(listed); ++thread) listed = shares[thread].failure;
    // Counts land one past their key, so their prefix sums are the starts; scattering moves
    // them one key on, and the closing shift moves them back.
    for (std::size_t thread = 0; thread != threads && succeeded(listed); ++thread)
        for (u32_t const position : shares[thread].positions) ++starts[position + 1];
    for (std::size_t position = 0; position != keys.size(); ++position) starts[position + 1] += starts[position];
    if (succeeded(listed) && failed(nodes.resize_uninitialized(starts[keys.size()]))) listed = status_t::bad_alloc_k;
    for (std::size_t thread = 0; thread != threads && succeeded(listed); ++thread)
        for (std::size_t hit = 0; hit != shares[thread].positions.size(); ++hit)
            nodes[starts[shares[thread].positions[hit]]++] = shares[thread].nodes[hit];
    std::destroy_n(shares, threads);
    if (failed(listed)) return listed;
    for (std::size_t position = keys.size(); position; --position) starts[position] = starts[position - 1];
    starts[0] = 0;
    for (std::size_t position = 0; position != keys.size(); ++position) {
        std::size_t const count = starts[position + 1] - starts[position];
        partial_sort(std::span<shard_node_t>(nodes.data() + starts[position], count), count, std::less<> {});
    }
    return status_t::success_k;
}

#pragma endregion Scans

#pragma region Reachability

/**
 *  @brief The two bitsets a reachability audit sweeps one part's nodes through, a bit per node in
 *      each, so a node's two bits are its state: 00 unreached, 10 in this level, 01 in the next
 *      one, and 11 once expanded.
 *
 *  Within a level only @c next_level changes, through relaxed atomics, as threads expand their
 *  nodes and reach others; @ref advance_level moves the next level in once every thread is done.
 */
template <template <typename> class atomic_ref_template_>
struct reachability_levels {

    static constexpr std::size_t lanes_k = std::numeric_limits<u64_t>::digits;

    /** The nodes of this level and of every earlier one, a bit per node. */
    std::span<u64_t> this_level;

    /** The nodes reached for the next level and those expanded so far, a bit per node. */
    std::span<u64_t> next_level;

    /** Clears both levels and puts @p landmarks nodes spread by stride over @p nodes in this one,
     *  at most one per node, answering how many it put. */
    std::size_t seed(std::size_t landmarks, std::size_t nodes) noexcept {
        std::memset(this_level.data(), 0, this_level.size_bytes());
        std::memset(next_level.data(), 0, next_level.size_bytes());
        std::size_t const seeds = min_of(landmarks, nodes);
        for (std::size_t seed = 0; seed != seeds; ++seed) {
            std::size_t const node = seed * nodes / seeds;
            this_level[node / lanes_k] |= u64_t {1} << (node % lanes_k);
        }
        return seeds;
    }

    /** Word @p word of the next level, which other threads may be setting. */
    u64_t later(std::size_t word) const noexcept {
        return atomic_ref_template_<u64_t>(next_level[word]).load(std::memory_order_relaxed);
    }

    /** The first node from @p from below @p bound this level holds unexpanded, else @p bound. */
    std::size_t first_in_level(std::size_t from, std::size_t bound) const noexcept {
        while (from < bound) {
            std::size_t const word = from / lanes_k;
            u64_t const lanes = (this_level[word] & ~later(word)) >> (from % lanes_k);
            if (lanes) return min_of<std::size_t>(from + std::countr_zero(lanes), bound);
            from = (word + 1) * lanes_k;
        }
        return bound;
    }

    /** Marks @p node expanded when this level holds it unexpanded, answering whether it was due. */
    bool expand(std::size_t node) const noexcept {
        std::size_t const word = node / lanes_k;
        u64_t const lane = u64_t {1} << (node % lanes_k);
        if (!(this_level[word] & ~later(word) & lane)) return false;
        atomic_ref_template_<u64_t>(next_level[word]).set_bits(lane, std::memory_order_relaxed);
        return true;
    }

    /** Puts @p neighbor in the next level unless some level already holds it. */
    void reach(std::size_t neighbor) const noexcept {
        std::size_t const word = neighbor / lanes_k;
        u64_t const lane = u64_t {1} << (neighbor % lanes_k);
        // The plain check spares the read-modify-write for the many neighbors already reached.
        if ((this_level[word] | later(word)) & lane) return;
        atomic_ref_template_<u64_t>(next_level[word]).set_bits(lane, std::memory_order_relaxed);
    }

    /** Moves the next level's newly reached nodes into this one, answering how many. */
    std::size_t advance_level() noexcept {
        std::size_t level_nodes = 0;
        for (std::size_t word = 0; word != this_level.size(); ++word) {
            u64_t const was = this_level[word], reached = next_level[word];
            level_nodes += static_cast<std::size_t>(std::popcount(reached & ~was));
            this_level[word] = was | reached;
            next_level[word] = was & reached;
        }
        return level_nodes;
    }

    /** The nodes any level reached. */
    std::size_t reachable() const noexcept {
        std::size_t reached = 0;
        for (u64_t const word : this_level) reached += static_cast<std::size_t>(std::popcount(word));
        return reached;
    }
};

/**
 *  @brief Sweeps one part of @p nodes nodes forward from @p landmarks stride-spread ones, level by
 *      level over @p this_level and @p next_level, each a bit per node at least, and answers how
 *      many nodes the sweep reached and in how many levels.
 *
 *  A level scans tasks of @p run_nodes nodes each through @c nvme_scan_reads, every engine's queue
 *  kept full: a task reads its nodes' neighbors in ID order, nodes whose blocks touch merged into
 *  one read that fits a staging buffer of @p buffer_bytes, so a full level reads near-sequentially
 *  and every reachable node is read once, by the level expanding it.
 *  @p neighbors_extent answers the @c nvme_extent_t on @p media holding a node's neighbor IDs.
 *  Runs inside a call @c nvme_begin_scans began. The sweep runs forward only, so it reports no
 *  @c back_sweeps at all.
 *
 *  @par Statuses:
 *  - @c overflow_k A read finds its engine's queue full.
 *  - @c io_k A neighbors read came back short.
 */
template <typename node_id_type_, template <typename> class atomic_ref_template_, typename executor_type_,
          typename neighbors_extent_type_>
expected<reachability_t> nvme_audit_reachability(executor_type_& executor,
                                                 std::span<typename executor_type_::medium_t const> media,
                                                 std::size_t buffer_bytes, std::span<u64_t> this_level,
                                                 std::span<u64_t> next_level, std::size_t nodes, std::size_t landmarks,
                                                 std::size_t run_nodes,
                                                 neighbors_extent_type_ const& neighbors_extent) noexcept {
    using levels_t = reachability_levels<atomic_ref_template_>;
    std::size_t const words = divide_round_up(nodes, levels_t::lanes_k);
    levels_t levels {this_level.first(words), next_level.first(words)};
    auto describe = [&](std::size_t task, u64_t& progress) noexcept -> std::array<nvme_extent_t, 1> {
        // A task's progress counts its nodes the reads issued so far cover.
        std::size_t const task_first = task * run_nodes, bound = min_of(task_first + run_nodes, nodes);
        for (std::size_t first = levels.first_in_level(task_first + progress, bound); first != bound;) {
            nvme_extent_t read = neighbors_extent(first);
            std::size_t last = first;
            for (std::size_t node = levels.first_in_level(first + 1, bound); node != bound;
                 node = levels.first_in_level(node + 1, bound)) {
                nvme_extent_t const extent = neighbors_extent(node);
                if (!nvme_extends_read(media[read.medium], read, extent, buffer_bytes)) break;
                last = node, read.length = static_cast<u32_t>(extent.begin + extent.length - read.begin);
            }
            progress = last + 1 - task_first;
            if (read.length) return {read};
            // Nodes without neighbors expand here, as no read carries them.
            for (std::size_t node = first; node <= last; ++node) levels.expand(node);
            first = levels.first_in_level(last + 1, bound);
        }
        return {};
    };
    auto consume = [&](std::size_t, std::size_t task, u64_t progress, std::array<nvme_extent_t, 1> const& reads,
                       std::array<byte_t const*, 1> const& fetched) noexcept {
        // A read holds the due nodes from its describe's start up to the first that ends past it.
        std::size_t const task_first = task * run_nodes, bound = min_of(task_first + run_nodes, nodes);
        for (std::size_t node = levels.first_in_level(task_first + progress, bound); node != bound;
             node = levels.first_in_level(node + 1, bound)) {
            nvme_extent_t const extent = neighbors_extent(node);
            if (extent.begin + extent.length > reads[0].begin + reads[0].length) break;
            levels.expand(node);
            std::span<node_id_type_ const> const neighbors {
                reinterpret_cast<node_id_type_ const*>(fetched[0] + (extent.begin - reads[0].begin)),
                extent.length / sizeof(node_id_type_)};
            for (node_id_type_ const neighbor : neighbors)
                if (neighbor < nodes) levels.reach(neighbor);
        }
    };

    reachability_t audited;
    audited.audited_nodes = nodes;
    std::size_t const tasks = divide_round_up(nodes, run_nodes);
    for (std::size_t level_nodes = levels.seed(landmarks, nodes); level_nodes; level_nodes = levels.advance_level()) {
        auto const [reads, status] = nvme_scan_reads<1, atomic_ref_template_>(executor, media, tasks, describe,
                                                                              consume);
        if (failed(status)) return {{}, status};
        ++audited.sweeps;
    }
    audited.reachable_nodes = levels.reachable();
    return {audited, status_t::success_k};
}

#pragma endregion Reachability

#pragma region Cursors

/**
 *  @brief A reader's walk over the entries of the parts an index was saved as, in the order a
 *      @c cursor_walk walks the RAM index they came from: every entry as stored, part after part,
 *      or breadth-first from the start entries, each entry once, each with the keys it links to.
 *
 *  Every call names the parts' reader, which must be the one the cursor opened on: its
 *  @c shards_count and @c part_size, and its @c find_first, @c read_keys and @c read_neighbors over
 *  @c shard_node locators, each through an executor. The parts are frozen, so the walk is pure
 *  position. A pull takes its entries from the walk in batches, reads their neighbors rows in one
 *  @c read_neighbors, visits the links in entry order, then reads the batch's keys and its links'
 *  keys through @c read_keys. A breadth-first walk holds one bit per entry and its frontier.
 */
template <typename key_type_, typename node_id_type_>
struct nvme_parts_cursor {
    using key_t = key_type_;
    using node_id_t = node_id_type_;
    using walk_t = cursor_walk<node_id_t>;
    using shard_node_t = typename walk_t::shard_node_t;

    /** Where a pull gathers its entries' links, in the caller's arena: the linked keys end to end,
     *  and the offset where each entry's run ends. */
    using links_allocator_t = arena_allocator<byte_t, forkunion_allocator_t>;
    using links_keys_t = growing_buffer<key_t, links_allocator_t>;
    using links_offsets_t = growing_buffer<std::size_t, links_allocator_t>;

    explicit nvme_parts_cursor(forkunion_allocator_t allocator) noexcept
        : starts_(allocator), walk_(allocator), entries_(allocator), offsets_(allocator), neighbors_(allocator),
          linked_(allocator) {}

    /**
     *  @brief Opens the walk over every part @p reader reads; any @p starts make it breadth-first,
     *      each key starting from its lowest node in the lowest part holding it, and a key held
     *      nowhere starting nothing.
     *
     *  Opening names no executor, so the cursor keeps @p starts and the first pull resolves them
     *  through one scan of the key columns.
     *
     *  @par Statuses:
     *  - @c bad_alloc_k The kept starts or the visited bits found no memory.
     */
    template <typename reader_type_>
    status_t open(reader_type_ const& reader, std::span<key_t const> starts) noexcept {
        if (starts.empty()) return status_t::success_k;
        if (failed(starts_.reserve_exactly(starts.size()))) return status_t::bad_alloc_k;
        for (key_t const& key : starts) starts_.push_back_reserved(key);
        return walk_.open_breadth_first(reader.shards_count(),
                                        [&](std::size_t part) noexcept { return reader.part_size(part); });
    }

    /**
     *  @brief Writes up to @p keys.size() next entries' keys into @p keys, gathering each one's
     *      links into @p links_keys and @p links_offsets when both are named, and answers the
     *      number of keys written.
     *
     *  @par Statuses:
     *  - @c bad_alloc_k A link or a newly visited entry found no memory; the batches written before
     *    stand.
     *  - @c contended_k Another call holds @p executor, or its pool is running another dispatch.
     *  - @c uninitialized_k @p executor holds no threads or pool workers.
     *  - @c config_mismatch_k A medium the engines cannot read.
     *  - @c overflow_k A staging region's byte count overflows, or a read finds its engine's queue
     *    full.
     *  - @c io_k A read came back short; the batches written before stand.
     */
    template <typename reader_type_, typename executor_type_>
    expected<std::size_t> pull(reader_type_ const& reader, std::span<key_t> keys, links_keys_t* links_keys,
                               links_offsets_t* links_offsets, executor_type_& executor) noexcept {
        if (!starts_.empty())
            if (status_t const resolved = visit_starts_(reader, executor); failed(resolved)) return {0, resolved};
        auto enter = [&](u32_t part) noexcept { return reader.part_size(part); };
        std::size_t written = 0;
        while (written != keys.size()) {
            std::size_t const wanted = keys.size() - written;
            entries_.clear();
            if (failed(entries_.reserve(wanted))) return {written, status_t::bad_alloc_k};
            shard_node_t entry;
            while (entries_.size() != wanted && walk_.next(entry, reader.shards_count(), enter))
                entries_.push_back_reserved(entry);
            if (entries_.empty()) break;
            status_t const linked = walk_.breadth_first() || links_keys
                                        ? follow_links_(reader, links_keys, links_offsets, executor)
                                        : status_t::success_k;
            if (failed(linked)) return {written, linked};
            if (status_t const read = reader.read_keys(to_span(entries_), keys.subspan(written, entries_.size()),
                                                       executor);
                failed(read))
                return {written, read};
            written += entries_.size();
        }
        return {written, status_t::success_k};
    }

  private:
    /** Visits the earliest entry of every kept start, in start order, then forgets the starts. */
    template <typename reader_type_, typename executor_type_>
    status_t visit_starts_(reader_type_ const& reader, executor_type_& executor) noexcept {
        if (failed(entries_.resize_uninitialized(starts_.size()))) return status_t::bad_alloc_k;
        if (status_t const found = reader.find_first(to_span(starts_), to_writable_span(entries_), executor);
            failed(found))
            return found;
        for (shard_node_t const start : entries_)
            if (start.shard != reader.shards_count())
                if (status_t const visited = walk_.visit(start.shard, start.node); failed(visited)) return visited;
        starts_.clear();
        return status_t::success_k;
    }

    /** Reads the batch's neighbors rows, visits their links in a breadth-first walk, and gathers
     *  them with their keys into @p links_keys and @p links_offsets when named. A self-loop marks
     *  a row with no neighbors, as the RAM cursors read it. */
    template <typename reader_type_, typename executor_type_>
    status_t follow_links_(reader_type_ const& reader, links_keys_t* links_keys, links_offsets_t* links_offsets,
                           executor_type_& executor) noexcept {
        if (failed(offsets_.resize_uninitialized(entries_.size() + 1))) return status_t::bad_alloc_k;
        if (status_t const read = reader.read_neighbors(to_span(entries_), to_writable_span(offsets_), neighbors_,
                                                        executor);
            failed(read))
            return read;
        std::size_t const links_before = links_keys ? links_keys->size() : 0;
        linked_.clear();
        if (links_keys && (failed(linked_.reserve(neighbors_.size())) ||
                           failed(links_offsets->reserve(links_offsets->size() + entries_.size()))))
            return status_t::bad_alloc_k;
        for (std::size_t position = 0; position != entries_.size(); ++position) {
            shard_node_t const from = entries_[position];
            node_id_t const part_size = reader.part_size(from.shard);
            for (std::size_t slot = offsets_[position]; slot != offsets_[position + 1]; ++slot) {
                node_id_t const neighbor = neighbors_[slot];
                if (neighbor >= part_size || neighbor == from.node) continue;
                status_t const visited = walk_.breadth_first() ? walk_.visit(from.shard, neighbor)
                                                               : status_t::success_k;
                if (failed(visited)) return visited;
                if (links_keys) linked_.push_back_reserved({from.shard, neighbor});
            }
            if (links_keys) links_offsets->push_back_reserved(links_before + linked_.size());
        }
        if (!links_keys) return status_t::success_k;
        // Extending keeps earlier batches' links; a resize drops them on growth.
        if (failed(links_keys->reserve(links_before + linked_.size()))) return status_t::bad_alloc_k;
        links_keys->extend_uninitialized(links_before + linked_.size());
        return reader.read_keys(to_span(linked_), std::span<key_t>(links_keys->data() + links_before, linked_.size()),
                                executor);
    }

    /** The start keys until the first pull resolves them; empty after. */
    growing_buffer<key_t, forkunion_allocator_t> starts_;

    /** The order of the walk and where it stands. */
    walk_t walk_;

    /** The entries of the batch a pull is reading, in walk order. */
    growing_buffer<shard_node_t, forkunion_allocator_t> entries_;

    /** Where each batch entry's neighbors start in @c neighbors_, then their end. */
    growing_buffer<std::size_t, forkunion_allocator_t> offsets_;

    /** The batch entries' neighbors rows end to end, sentinels included. */
    growing_buffer<node_id_t, forkunion_allocator_t> neighbors_;

    /** The batch's links below their parts' sizes, in entry order, whose keys a pull reads. */
    growing_buffer<shard_node_t, forkunion_allocator_t> linked_;
};

#pragma endregion Cursors

#pragma region Sampling

/**
 *  @brief What every NVMe tier's SAMPLE source reads beside its scores: the shown and focus keys'
 *      entries, the keys of nodes already read, neighbors rows, and blocking rounds on one engine.
 *
 *  Each tower's source owns one, names its own rows' extents, and scores; see @c sample_source.
 *  Every read lands in the engine's staging buffers, valid until the next read. A failed read
 *  lands in @c failure, and every later row read answers no row.
 */
template <typename part_type_, typename key_type_, typename node_id_type_, typename engine_type_>
struct nvme_sample_rows {
    using key_t = key_type_;
    using node_id_t = node_id_type_;
    using shard_node_t = shard_node<node_id_t>;
    using medium_t = typename engine_type_::medium_t;

    /** Every distinct shown and focus key filed under its listed position, in thread zero's
     *  arena. */
    using listed_positions_t =
        chunked_multimap<key_t, u32_t, std::hash<key_t>, arena_allocator<byte_t, forkunion_allocator_t>>;

    /** Thread zero's engine, which the plan reads through, and the media it holds attached. */
    engine_type_& engine;
    std::span<medium_t const> media;

    /** Every part, for where its key column starts and how its keys are stored. */
    std::span<part_type_ const> parts;
    std::span<node_id_t const> watermarks_snapshot;

    /** The distinct shown and focus keys' positions, and where each position's entries start. */
    listed_positions_t const& listed_positions;
    std::span<std::size_t const> listed_starts;
    std::span<shard_node_t const> listed_nodes;

    /** Keys a round fetched beside their rows, direct-mapped by node. */
    std::span<shard_node_t> cached_nodes;
    std::span<key_t> cached_keys;

    /** One neighbors row, copied out of the staging buffers before its callbacks score. */
    std::span<node_id_t> staged_neighbors;

    /** One scoring round's reads and where each landed. */
    std::span<nvme_extent_t> round_extents;
    std::span<byte_t const*> round_bytes;
    status_t* failure;

    key_t key_of(shard_node_t node) const noexcept {
        std::size_t const slot = cache_slot_(node);
        if (cached_nodes[slot] == node) return cached_keys[slot];
        bytes_view_t const fetched = read_row(key_extent_(node));
        if (fetched.empty()) return key_t {};
        key_t const key = safegraphs_widened_key<key_t>(fetched.data(), parts[node.shard].key_dtype);
        cached_nodes[slot] = node, cached_keys[slot] = key;
        return key;
    }

    /** Answers only the shown and focus keys, which are all the plan asks for. */
    template <typename callback_type_>
    void nodes_of(key_t const& key, callback_type_&& callback) const noexcept {
        u32_t const listed_count = static_cast<u32_t>(listed_starts.size() - 1);
        listed_positions.find_each(key, listed_count, [&](u32_t listed) noexcept {
            for (std::size_t entry = listed_starts[listed]; entry != listed_starts[listed + 1]; ++entry)
                callback(listed_nodes[entry]);
        });
    }

    /** The neighbors row at @p extent, copied out first, since a callback may score through the
     *  staging buffers; empty after a failure. */
    std::span<node_id_t const> neighbors_row(nvme_extent_t extent) const noexcept {
        bytes_view_t const fetched = read_row(extent);
        if (!fetched.empty()) std::memcpy(staged_neighbors.data(), fetched.data(), fetched.size());
        return staged_neighbors.first(fetched.size() / sizeof(node_id_t));
    }

    /** The row at @p extent through a one-extent round into staging buffer zero; an empty row
     *  reads nothing, and a failure answers no row. */
    bytes_view_t read_row(nvme_extent_t extent) const noexcept {
        if (!extent.length || failed(*failure)) return {};
        round_extents[0] = extent;
        if (status_t const read = nvme_read_round(engine, media, round_extents.first(1), round_bytes.first(1));
            failed(read))
            return fail(read, {}), bytes_view_t {};
        return {round_bytes[0], extent.length};
    }

    /** Reads one scoring round: the query's row and each node's row of @p round, whose extents
     *  the source wrote from zero, then each node's key. A failure answers infinity for
     *  @p distances. */
    status_t read_round(std::span<shard_node_t const> round, std::span<f32_t> distances) const noexcept {
        for (std::size_t lane = 0; lane != round.size(); ++lane)
            round_extents[1 + round.size() + lane] = key_extent_(round[lane]);
        std::size_t const reads = 2 * round.size() + 1;
        status_t const read = nvme_read_round(engine, media, round_extents.first(reads), round_bytes.first(reads));
        if (failed(read)) fail(read, distances);
        return read;
    }

    /** Caches the keys the last @ref read_round fetched for @p round. */
    void remember_keys(std::span<shard_node_t const> round) const noexcept {
        for (std::size_t lane = 0; lane != round.size(); ++lane) {
            std::size_t const slot = cache_slot_(round[lane]);
            cached_nodes[slot] = round[lane];
            cached_keys[slot] = safegraphs_widened_key<key_t>(round_bytes[1 + round.size() + lane],
                                                              parts[round[lane].shard].key_dtype);
        }
    }

    /** Keeps the first failure, and answers infinity for @p distances. */
    void fail(status_t status, std::span<f32_t> distances) const noexcept {
        if (succeeded(*failure)) *failure = status;
        for (f32_t& distance : distances) distance = std::numeric_limits<f32_t>::infinity();
    }

  private:
    std::size_t cache_slot_(shard_node_t node) const noexcept {
        return mix_bits(u64_t {node.shard} << 32 ^ node.node) & (cached_nodes.size() - 1);
    }

    nvme_extent_t key_extent_(shard_node_t node) const noexcept {
        part_type_ const& part = parts[node.shard];
        return {part.keys_begin + u64_t {node.node} * part.key_row_bytes, part.key_row_bytes,
                static_cast<io_medium_id_t>(node.shard)};
    }
};

/**
 *  @brief Opens a SAMPLE call over @p parts on @p media as every NVMe tier opens one, lists the
 *      shown and focus keys' entries, and hands what the plan reads to @p draw.
 *
 *  The call holds the media with every engine at least a round deep, a round reading the query's
 *  row, then a row and a key per node of a neighbors row @p width wide, each into a staging buffer
 *  of @p buffer_bytes. Thread zero's arena holds the listed keys and their entries, the keys cached
 *  for as many visits as @p draws make under @p config, and one round's extents. @p scan_keys runs
 *  as <tt>scan_keys(keys, on_hit) noexcept → status_t</tt>, just as @c nvme_list_entries scans, and
 *  @p draw as <tt>draw(rows) noexcept → expected<sample_result_t></tt> while the media stay held.
 *
 *  @par Statuses:
 *  - @c uninitialized_k No part is viewed, or @p executor holds no threads or pool workers.
 *  - @c bad_alloc_k The arenas cannot hold the scratch.
 *  - @c contended_k Another call holds @p executor, or its pool is running another dispatch.
 *  - @c config_mismatch_k A medium the engines cannot read.
 */
template <typename key_type_, typename node_id_type_, typename part_type_, typename executor_type_,
          typename scan_keys_type_, typename draw_type_>
expected<sample_result_t> nvme_sample(executor_type_& executor,
                                      std::span<typename executor_type_::medium_t const> media,
                                      std::span<part_type_ const> parts, std::size_t width, std::size_t buffer_bytes,
                                      sample_inputs<key_type_> inputs, std::size_t draws, sample_config_t config,
                                      scan_keys_type_&& scan_keys, draw_type_&& draw) noexcept {
    using rows_t = nvme_sample_rows<part_type_, key_type_, node_id_type_, typename executor_type_::engine_t>;
    using shard_node_t = shard_node<node_id_type_>;
    using scratch_allocator_t = arena_allocator<byte_t, forkunion_allocator_t>;
    if (parts.empty()) return {{}, status_t::uninitialized_k};
    std::size_t const round_reads = 2 * width + 1;
    std::size_t const queue_depth = max_of(nvme_call_depth(executor, media, parts), round_reads);
    auto [call, call_status] = nvme_begin_call(executor, media, queue_depth, 0, round_reads, buffer_bytes);
    if (failed(call_status)) return {{}, call_status};

    scratch_allocator_t const scratch {executor.arena(0)};
    std::size_t const listed_count = inputs.shown.size() + inputs.focus.size();
    std::size_t const visits = draws * (config.expansion_factor ? config.expansion_factor : sample_expansion_factor_k);
    // A region's visits are the widest batch the plan scores before reading their keys.
    std::size_t const cached_count = std::bit_ceil(max_of<std::size_t>(visits, 1));
    typename rows_t::listed_positions_t listed_positions {scratch};
    growing_buffer<key_type_, scratch_allocator_t> listed_keys {scratch}, cached_keys {scratch};
    growing_buffer<std::size_t, scratch_allocator_t> listed_starts {scratch};
    growing_buffer<shard_node_t, scratch_allocator_t> listed_nodes {scratch}, cached_nodes {scratch};
    growing_buffer<node_id_type_, scratch_allocator_t> watermarks {scratch}, staged_neighbors {scratch};
    growing_buffer<nvme_extent_t, scratch_allocator_t> round_extents {scratch};
    growing_buffer<byte_t const*, scratch_allocator_t> round_bytes {scratch};
    if (failed(listed_keys.reserve_exactly(listed_count)) || failed(listed_positions.reserve(listed_count)) ||
        failed(listed_starts.resize(listed_count + 1)) || failed(cached_keys.resize_uninitialized(cached_count)) ||
        failed(cached_nodes.resize_uninitialized(cached_count)) ||
        failed(watermarks.resize_uninitialized(parts.size())) || failed(staged_neighbors.resize_uninitialized(width)) ||
        failed(round_extents.resize(round_reads)) || failed(round_bytes.resize(round_reads)))
        return {{}, status_t::bad_alloc_k};
    for (std::size_t part = 0; part != parts.size(); ++part) watermarks[part] = parts[part].nodes;
    for (shard_node_t& cached : cached_nodes) cached = {static_cast<u32_t>(parts.size()), 0};
    for (key_type_ const& key : inputs.shown) listed_keys.push_back_reserved(key);
    for (key_type_ const& key : inputs.focus) listed_keys.push_back_reserved(key);
    // A key shown and focused, or repeated, is listed once, so each lookup finds one position.
    u32_t distinct = 0;
    for (key_type_ const& key : listed_keys)
        if (!listed_positions.contains(key, distinct))
            listed_positions.insert(key, distinct), listed_keys[distinct++] = key;
    std::span<std::size_t> const distinct_starts = to_writable_span(listed_starts).first(distinct + 1u);
    if (status_t const listed = nvme_list_entries(to_span(listed_keys).first(distinct), distinct_starts, listed_nodes,
                                                  executor, scan_keys);
        failed(listed))
        return {{}, listed};

    status_t failure = status_t::success_k;
    return draw(rows_t {executor.engine(0), media, parts, to_span(watermarks), listed_positions, distinct_starts,
                        to_span(listed_nodes), to_writable_span(cached_nodes), to_writable_span(cached_keys),
                        to_writable_span(staged_neighbors), to_writable_span(round_extents),
                        to_writable_span(round_bytes), &failure});
}

/**
 *  @brief Draws and places through a @c sample_plan over @p source on executor thread zero, the
 *      pool thread whose engine the source reads through, then counts the draws' ordinals; the
 *      @p failure the source's reads kept answers last.
 *
 *  @par Statuses:
 *  - @c bad_alloc_k Thread zero's arena cannot hold the plan's scratch.
 *  - @c shape_mismatch_k An output's or the shown coordinates' length disagrees.
 *  - @c io_k A read came back short; the draws written stand, their later reads unscored.
 */
template <typename source_type_, typename key_type_, typename executor_type_>
expected<sample_result_t> nvme_draw_sample(source_type_ const& source, sample_inputs<key_type_> inputs,
                                           sample_outputs<key_type_> outputs, sample_config_t config,
                                           executor_type_& executor, status_t const& failure) noexcept {
    using plan_t = sample_plan<source_type_>;
    std::size_t const plan_bytes = plan_t::scratch_bytes(outputs.keys.size(), inputs.shown.size(), config);
    byte_t* const plan_scratch = executor.arena(0).allocate(plan_bytes, default_alignment_k);
    if (!plan_scratch) return {{}, status_t::bad_alloc_k};
    plan_t plan {source, inputs, outputs, config, {plan_scratch, plan_bytes}};
    // Thread zero's engine reads for the plan and serves only the pool thread that opened it.
    status_t status = status_t::success_k;
    status_t const dispatched = executor.for_threads([&](std::size_t thread, std::size_t) noexcept {
        if (thread) return;
        status = plan.draw();
        if (succeeded(status)) status = plan.assign_coordinates();
    });
    if (failed(dispatched)) status = dispatched;
    if (succeeded(status)) status = plan.assign_ordinals();
    if (succeeded(status)) status = failure;
    return {plan.result(), status};
}

#pragma endregion Sampling

} // namespace unum::usearch::utilities
