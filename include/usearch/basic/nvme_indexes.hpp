/**
 *  @file include/usearch/basic/nvme_indexes.hpp
 *  @author Ash Vardanian
 *  @date October 14, 2026
 *  @brief Similarity Search over the parts a basic graph was saved as, on block storage, read-only:
 *      the sharded index of @c nvme_index, serving a SafeGraphs monolith or manifest as the
 *      compiled library saves them.
 *
 *  A view sniffs a monolith or a manifest by content, turns every part's SafeGraphs header into the
 *  @c persisted_columns_t its format's producer fills, and opens the parts on the executor's
 *  threads. Every call walks every part of a query on one thread through the walks of
 *  @c nvme_parts_catalog, merging the parts' rankings by key.
 */
#pragma once

#include <cstring> // `std::memcmp`

#include <array>       // `std::array` - the monolith magic a view sniffs
#include <concepts>    // `std::same_as`
#include <span>        // `std::span`
#include <string_view> // `std::string_view`
#include <utility>     // `std::forward`, `std::move`

#include <forkunion/atomics.hpp> // `fu::standard_atomic_ref` - the default reference

#include "usearch/utilities/types.hpp"
#include "usearch/utilities/blocking_files.hpp"    // `seekable_file_t` - the magic a view sniffs
#include "usearch/utilities/nvme_walks.hpp"        // `nvme_read_header`, `nvme_parts_cursor` - the cursor's walk
#include "usearch/utilities/ram_buffers.hpp"       // `growing_buffer`, `arena_allocator`
#include "usearch/utilities/safegraphs_format.hpp" // `safegraphs_persisted_part`, `safegraphs_read_manifest`
#include "usearch/basic/nvme_index.hpp"

namespace unum::usearch::basic {

/**
 *  @brief Similarity Search over the parts a basic RAM index saved, on block storage, read-only.
 *
 *  The class is a catalog of parts: each part's medium, its columns' offsets and the directories of
 *  its ragged columns, and nothing per node beyond those few bits a row. Every call borrows an
 *  executor, attaches the media to its engines for its own duration, and walks every part of a
 *  query on one thread, merging the parts' rankings by key, through the walks of
 *  @c nvme_parts_catalog.
 *
 *  It keeps @p node_id_type_ and @p key_type_ as free as @c nvme_index does; only the compiled
 *  library narrows them, to the 32-bit IDs its sharded indexes write.
 */
template <typename codec_type_, typename key_type_ = uuid128_t, typename node_id_type_ = u32_t,
          typename distance_type_ = f64_t, typename engine_type_ = default_io_engine<dummy_waiter_t>,
          typename row_kit_type_ = st::serial_row_kit_t,
          template <typename> class atomic_ref_template_ = fu::standard_atomic_ref,
          typename wait_type_ = dummy_waiter_t>
    requires std::same_as<node_id_type_, u32_t> || std::same_as<node_id_type_, u64_t>
class nvme_indexes {
    template <typename>
    friend struct nvme_indexes_cursor;

  public:

    /** The walks every call runs over every part. */
    using catalog_t = nvme_parts_catalog<codec_type_, key_type_, node_id_type_, distance_type_, engine_type_,
                                         row_kit_type_, atomic_ref_template_, wait_type_>;
    template <typename word_type_>
    using atomic_ref = typename catalog_t::template atomic_ref<word_type_>;

    using codec_t = typename catalog_t::codec_t;
    using key_t = typename catalog_t::key_t;
    using distance_t = typename catalog_t::distance_t;
    using engine_t = typename catalog_t::engine_t;
    using executor_t = typename catalog_t::executor_t;
    using medium_t = typename catalog_t::medium_t;
    using arena_t = typename catalog_t::arena_t;
    using node_id_t = typename catalog_t::node_id_t;
    using scored_node_t = typename catalog_t::scored_node_t;
    using scan_payload_t = typename catalog_t::scan_payload_t;
    using walk_payload_t = typename catalog_t::walk_payload_t;
    using search_config_t = typename catalog_t::search_config_t;
    using search_result_t = typename catalog_t::search_result_t;
    using reachability_t = typename catalog_t::reachability_t;
    using shard_node_t = typename catalog_t::shard_node_t;

    /** The reader's walk over the entries, as stored or breadth-first from start keys. */
    using cursor_t = nvme_indexes_cursor<nvme_indexes>;

    nvme_indexes() = delete;
    explicit nvme_indexes(forkunion_allocator_t allocator) noexcept
        : media_(allocator), parts_(allocator), allocator_(allocator) {}
    nvme_indexes(nvme_indexes&&) noexcept = default;
    nvme_indexes& operator=(nvme_indexes&&) noexcept = default;
    nvme_indexes(nvme_indexes const&) = delete;
    nvme_indexes& operator=(nvme_indexes const&) = delete;

    /**
     *  The checkpoint's metadata without viewing it, @ref nvme_read_metadata.
     *
     *  @par Statuses:
     *  - @c invalid_argument_k @p path, or a manifest's first part path, does not parse or names an
     *    unusable base or namespace.
     *  - @c unsupported_k A transport ID in a build without SPDK.
     *  - @c malformed_k A transport ID starting no monolith, or a manifest or header that does not
     *    parse.
     */
    static expected<safegraphs_metadata_t> metadata(std::string_view path) noexcept {
        return nvme_read_metadata<wait_type_>(path);
    }

    /**
     *  @brief Opens a manifest or a monolith, whichever its content names, with every part's
     *      header, medium and row directories, refusing parts that disagree with this
     *      instantiation. A failed view leaves the index empty.
     *
     *  @param[in] path A monolith, or the manifest whose parts sit beside it.
     *  @param[in] paths Where to serve each part from instead, one per part: a directory holding
     *      the part's file, or the file itself; empty serves every part where the manifest points.
     *  @param[in] metric The token every part must name, or empty to take any.
     *  @param[in,out] executor Builds the parts' directories, one part per thread at a time.
     *
     *  @par Statuses:
     *  - @c invalid_argument_k @p path is empty, or it or a part's path does not parse or names a
     *    base off the medium's block.
     *  - @c unsupported_k @p path is a transport ID.
     *  - @c malformed_k The manifest names no part, or a manifest, header or row directory does not
     *    parse.
     *  - @c config_mismatch_k @p paths does not name one place per part, or a part disagrees with
     *    this instantiation's codec, keys, node IDs or @p metric, or lays out a graph no reader
     *    walks.
     *  - @c contended_k Another call holds @p executor, or its pool is running another dispatch.
     *  - @c uninitialized_k @p executor holds no threads or pool workers.
     *  - @c overflow_k A header read spans more than one read can move.
     */
    status_t view(std::string_view path, std::span<std::string_view const> paths, std::string_view metric,
                  executor_t& executor) noexcept {
        media_.clear();
        parts_.clear();
        status_t const viewed = view_(path, paths, metric, executor);
        if (failed(viewed)) media_.clear(), parts_.clear();
        return viewed;
    }

    /** Shards in the view, one per part of the checkpoint, so a monolith views as one shard. */
    std::size_t shards_count() const noexcept { return parts_.size(); }

    /** Entries across every shard. */
    std::size_t size() const noexcept { return catalog_().size(); }

    /** Entries the view hosts, which are exactly those its parts hold, since a view never grows. */
    std::size_t capacity() const noexcept { return size(); }

    /** Value elements across every part, as @ref nvme_parts_catalog::value_elements counts them. */
    std::size_t value_elements() const noexcept { return catalog_().value_elements(); }

    /** Neighbor IDs of the widest neighbors row across every part. */
    std::size_t connectivity() const noexcept { return catalog_().connectivity(); }

    /** Bytes the index occupies: the catalog and the row directories, a few bits a row. */
    std::size_t memory_usage() const noexcept {
        std::size_t bytes = sizeof(nvme_indexes) + media_.memory_usage() + parts_.memory_usage();
        for (part_t const& part : parts_) bytes += part.value_starts.size_bytes() + part.neighbor_starts.size_bytes();
        return bytes;
    }

    /** The walk breadth @p config asks for, or twice the widest neighbors row when it asks for
     *  none. */
    std::size_t non_zero_expansion_factor(search_config_t const& config) const noexcept {
        return catalog_().non_zero_expansion_factor(config);
    }

    /**
     *  @ref nvme_parts_catalog::search over every part, merging the parts' rankings by key.
     *
     *  @par Statuses:
     *  - @c uninitialized_k Nothing is viewed, or the executor holds no threads or pool workers.
     *  - @c config_mismatch_k The counts are not one per query, the keys and distances differ in
     *    length or do not split into one run per query, or a medium the engines cannot read.
     *  - @c contended_k Another call holds the executor, or its pool is running another dispatch.
     *  - @c overflow_k A staging region's byte count overflows.
     */
    template <typename... arguments_type_>
    expected<search_result_t> search(arguments_type_&&... arguments) noexcept {
        return catalog_().search(std::forward<arguments_type_>(arguments)...);
    }

    /**
     *  @ref nvme_parts_catalog::scan over every part.
     *
     *  @par Statuses:
     *  - @c uninitialized_k Nothing is viewed, or the executor holds no threads or pool workers.
     *  - @c config_mismatch_k The counts are not one per query, the keys and distances differ in
     *    length or do not split into one run per query, or a medium the engines cannot read.
     *  - @c contended_k Another call holds the executor, or its pool is running another dispatch.
     *  - @c overflow_k A staging region's byte count overflows, or a read finds its engine's queue
     *    full.
     */
    template <typename... arguments_type_>
    expected<search_result_t> scan(arguments_type_&&... arguments) noexcept {
        return catalog_().scan(std::forward<arguments_type_>(arguments)...);
    }

    /**
     *  @ref nvme_parts_catalog::scan_within over every part.
     *
     *  @par Statuses:
     *  - @c uninitialized_k Nothing is viewed, or the executor holds no threads or pool workers.
     *  - @c contended_k Another call holds the executor, or its pool is running another dispatch.
     *  - @c overflow_k A staging region's byte count overflows, or a read finds its engine's queue
     *    full.
     *  - @c config_mismatch_k A medium the engines cannot read.
     */
    template <typename... arguments_type_>
    expected<search_result_t> scan_within(arguments_type_&&... arguments) noexcept {
        return catalog_().scan_within(std::forward<arguments_type_>(arguments)...);
    }

    /**
     *  @ref nvme_parts_catalog::count over every part.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k @p entries and @p keys differ in length, or a medium the engines
     *    cannot read.
     *  - @c uninitialized_k Nothing is viewed, or @p executor holds no threads or pool workers.
     *  - @c contended_k Another call holds @p executor, or its pool is running another dispatch.
     *  - @c overflow_k A staging region's byte count overflows, or a read finds its engine's queue
     *    full.
     */
    status_t count(std::span<key_t const> keys, std::span<std::size_t> entries, executor_t& executor) const noexcept {
        return catalog_().count(keys, entries, executor);
    }

    /**
     *  @ref nvme_parts_catalog::contains over every part.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k The entries and the keys differ in length, or a medium the engines
     *    cannot read.
     *  - @c uninitialized_k Nothing is viewed, or the executor holds no threads or pool workers.
     *  - @c contended_k Another call holds the executor, or its pool is running another dispatch.
     *  - @c overflow_k A staging region's byte count overflows, or a read finds its engine's queue
     *    full.
     */
    template <typename... arguments_type_>
    status_t contains(arguments_type_&&... arguments) const noexcept {
        return catalog_().contains(std::forward<arguments_type_>(arguments)...);
    }

    /**
     *  @ref nvme_parts_catalog::get over every part.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k The entries and the keys differ in length, or a medium the engines
     *    cannot read.
     *  - @c uninitialized_k Nothing is viewed, or the executor holds no threads or pool workers.
     *  - @c contended_k Another call holds the executor, or its pool is running another dispatch.
     *  - @c overflow_k A staging region's byte count overflows, or a read finds its engine's queue
     *    full.
     */
    template <typename... arguments_type_>
    status_t get(arguments_type_&&... arguments) const noexcept {
        return catalog_().get(std::forward<arguments_type_>(arguments)...);
    }

    /**
     *  @ref nvme_parts_catalog::read_entries over every part, in part order.
     *
     *  @par Statuses:
     *  - @c uninitialized_k Nothing is viewed, or the executor holds no threads or pool workers.
     *  - @c config_mismatch_k The entries run past @ref size, or a medium the engines cannot read.
     *  - @c contended_k Another call holds the executor, or its pool is running another dispatch.
     *  - @c overflow_k A staging region's byte count overflows, or a read finds its engine's queue
     *    full.
     */
    template <typename... arguments_type_>
    status_t read_entries(arguments_type_&&... arguments) const noexcept {
        return catalog_().read_entries(std::forward<arguments_type_>(arguments)...);
    }

    /**
     *  @ref nvme_parts_catalog::health over every part: a forward sweep from as many landmarks as
     *  one search draws, so @c back_sweeps stays zero.
     *
     *  @par Statuses:
     *  - @c uninitialized_k Nothing is viewed, or @p executor holds no threads or pool workers.
     *  - @c contended_k Another call holds @p executor, or its pool is running another dispatch.
     *  - @c overflow_k A staging region's byte count overflows, or a read finds its engine's queue
     *    full.
     *  - @c config_mismatch_k A medium the engines cannot read.
     *  - @c io_k A neighbors read came back short.
     */
    expected<reachability_t> health(executor_t& executor) const noexcept { return catalog_().health(executor); }

    /**
     *  @ref nvme_parts_catalog::sample over every part, for 32-bit node IDs.
     *
     *  @par Statuses:
     *  - @c uninitialized_k Nothing is viewed, or the executor holds no threads or pool workers.
     *  - @c shape_mismatch_k An output's or the shown coordinates' length disagrees.
     *  - @c bad_alloc_k The arenas cannot hold the scratch.
     *  - @c contended_k Another call holds the executor, or its pool is running another dispatch.
     *  - @c config_mismatch_k A medium the engines cannot read.
     *  - @c io_k A read came back short.
     */
    template <typename... arguments_type_>
    expected<sample_result_t> sample(arguments_type_&&... arguments) const noexcept {
        return catalog_().sample(std::forward<arguments_type_>(arguments)...);
    }

  private:
    using part_t = typename catalog_t::part_t;
    using scratch_allocator_t = arena_allocator<byte_t, forkunion_allocator_t>;

    /** The catalog over this index's media and parts, for one call. */
    catalog_t catalog_() const noexcept { return catalog_t {to_span(media_), to_span(parts_)}; }

    /** Opens the SafeGraphs part at @p path once: its header, read through a blocking read of the
     *  file, turned into the columns its producer fills and checked against this instantiation's
     *  codec, keys, node IDs and @p metric, unless empty, then the part over the same file. */
    template <typename chunk_allocator_type_>
    status_t open_safegraphs_part_(std::string_view path, std::string_view metric,
                                   growing_buffer<u64_t, chunk_allocator_type_>& chunk, medium_t& medium,
                                   part_t& part) const noexcept {
        auto [file, open_status] = file_medium_t::open(path);
        if (failed(open_status)) return open_status;
        growing_buffer<byte_t, forkunion_allocator_t> header_bytes {allocator_};
        if (status_t const read = nvme_read_header(file, header_bytes); failed(read)) return read;
        auto const [persisted, persisted_status] = safegraphs_persisted_part(to_span(header_bytes), allocator_);
        if (failed(persisted_status)) return persisted_status;
        persisted_columns_t const& columns = persisted.columns;
        bool const values_fixed = !columns.values.offsets;
        if (columns.values.dtype != codec_t::value_dtype_k ||
            persisted.value_shape != safegraphs_value_shape<codec_t>() ||
            values_fixed != safegraphs_value_is_fixed<codec_t>() ||
            !safegraphs_key_widens(columns.keys.dtype, safegraphs_key_dtype<key_t>()) ||
            columns.neighbors.dtype != safegraphs_node_id_dtype<node_id_t>() ||
            (!metric.empty() && !persisted.metric.empty() && persisted.metric.view() != metric))
            return status_t::config_mismatch_k;
        return catalog_t::open_part_(std::move(file), columns, chunk, allocator_, medium, part);
    }

    /** The body of @ref view, into an empty catalog: every part's path resolves on the calling
     *  thread, then the parts open on the executor's threads, one at a time per thread. */
    status_t view_(std::string_view checkpoint, std::span<std::string_view const> paths, std::string_view metric,
                   executor_t& executor) noexcept {
        if (checkpoint.empty()) return status_t::invalid_argument_k;
        if (names_transport_id(checkpoint)) return status_t::unsupported_k;
        std::array<byte_t, safegraphs_prefix_t::magic_k.size()> magic {};
        {
            auto [file, open_status] = file_medium_t::open(checkpoint);
            if (failed(open_status)) return open_status;
            // A manifest shorter than the magic reads short; parsing it below reports that.
            [[maybe_unused]] status_t const sniffed = file.read_at(0, magic);
        }
        // The magic makes a lone monolith, anything else a manifest naming its parts beside it.
        growing_buffer<char, forkunion_allocator_t> names {allocator_};
        growing_buffer<iota_view<std::size_t>, forkunion_allocator_t> entries {allocator_};
        if (std::memcmp(magic.data(), safegraphs_prefix_t::magic_k.data(), magic.size()) == 0) {
            if (failed(entries.push_back(iota_view<std::size_t> {0, 0}))) return status_t::bad_alloc_k;
        }
        else {
            status_t kept = status_t::success_k;
            auto per_part = [&](safegraphs_part_t entry) noexcept {
                std::size_t const first = names.size();
                for (char const character : entry.path)
                    if (failed(names.push_back(character))) kept = status_t::bad_alloc_k;
                if (failed(entries.push_back(iota_view<std::size_t> {first, names.size()})))
                    kept = status_t::bad_alloc_k;
            };
            auto const [summary, summary_status] = safegraphs_read_manifest(checkpoint, allocator_, per_part);
            if (failed(summary_status)) return summary_status;
            if (failed(kept)) return kept;
            if (entries.empty()) return status_t::malformed_k;
        }
        if (!paths.empty() && paths.size() != entries.size()) return status_t::config_mismatch_k;

        // Every resolved path goes into one buffer, terminated for the OS, so the threads read them
        // as they are.
        std::string_view const checkpoint_name = safegraphs_basename(checkpoint);
        std::string_view const checkpoint_directory = checkpoint.substr(0, checkpoint.size() - checkpoint_name.size());
        growing_buffer<char, forkunion_allocator_t> resolved {allocator_};
        growing_buffer<char, forkunion_allocator_t> resolved_paths {allocator_};
        growing_buffer<iota_view<std::size_t>, forkunion_allocator_t> resolved_entries {allocator_};
        if (failed(resolved_entries.reserve_exactly(entries.size()))) return status_t::bad_alloc_k;
        for (std::size_t part = 0; part != entries.size(); ++part) {
            std::string_view const relative = entries[part].empty()
                                                  ? checkpoint_name
                                                  : std::string_view {names.data() + entries[part].first,
                                                                      entries[part].size()};
            status_t joined = status_t::success_k;
            if (paths.empty())
                joined = entries[part].empty() ? nvme_join_path_(resolved, {}, checkpoint)
                                               : nvme_join_path_(resolved, checkpoint_directory, relative);
            else {
                // The directory test runs on the terminated copy, which the OS reads.
                joined = nvme_join_path_(resolved, {}, paths[part]);
                if (succeeded(joined) && nvme_is_directory_(resolved.data()))
                    joined = nvme_join_path_(resolved, paths[part], safegraphs_basename(relative));
            }
            if (failed(joined)) return joined;
            std::size_t const first = resolved_paths.size();
            for (char const character : resolved)
                if (failed(resolved_paths.push_back(character))) return status_t::bad_alloc_k;
            resolved_entries.push_back_reserved(iota_view<std::size_t> {first, resolved_paths.size() - 1});
        }

        if (failed(media_.resize(entries.size())) || failed(parts_.resize(entries.size())))
            return status_t::bad_alloc_k;
        auto [claim, claim_status] = executor.claim();
        if (failed(claim_status)) return claim_status;
        if (status_t const reserved = executor.reserve(catalog_t::scan_chunk_bytes_k + default_alignment_k, 0, 0);
            failed(reserved))
            return reserved;
        alignas(default_alignment_k) std::size_t cursor = 0;
        status_t failure = status_t::success_k;
        auto open_parts = [&](std::size_t thread, std::size_t) noexcept {
            growing_buffer<u64_t, scratch_allocator_t> chunk {scratch_allocator_t {executor.arena(thread)}};
            status_t opened = chunk.resize_uninitialized(catalog_t::scan_chunk_bytes_k / sizeof(u64_t));
            while (succeeded(opened) && succeeded(atomic_ref<status_t>(failure).load(std::memory_order_relaxed))) {
                std::size_t const part = atomic_ref<std::size_t>(cursor).fetch_add(1, std::memory_order_relaxed);
                if (part >= parts_.size()) return;
                std::string_view const path {resolved_paths.data() + resolved_entries[part].first,
                                             resolved_entries[part].size()};
                opened = open_safegraphs_part_(path, metric, chunk, media_[part], parts_[part]);
            }
            keep_first_failure<atomic_ref>(failure, opened);
        };
        if (status_t const fanned = executor.for_threads(open_parts); failed(fanned)) return fanned;
        return failure;
    }

    /** Every part's file, open for the engines, in part order. */
    growing_buffer<medium_t, forkunion_allocator_t> media_;

    /** Where every part keeps its columns and its row directories, parallel to @c media_. */
    growing_buffer<part_t, forkunion_allocator_t> parts_;

    /** The host allocator behind the catalog, the directories and the paths a view resolves. */
    forkunion_allocator_t allocator_;
};

/** A reader's walk over the entries of one @c nvme_indexes or @c nvme_index, in the order
 *  @c ram_indexes_cursor walks the RAM index the parts were saved from: a @c nvme_parts_cursor over
 *  the index's catalog. Every call names the index it walks, which must be the one it opened on. */
template <typename indexes_type_>
struct nvme_indexes_cursor {
    using indexes_t = indexes_type_;
    using key_t = typename indexes_t::key_t;
    using executor_t = typename indexes_t::executor_t;
    using walk_t = nvme_parts_cursor<key_t, typename indexes_t::node_id_t>;
    using links_keys_t = typename walk_t::links_keys_t;
    using links_offsets_t = typename walk_t::links_offsets_t;

    explicit nvme_indexes_cursor(forkunion_allocator_t allocator) noexcept : walk_(allocator) {}

    /** @ref nvme_parts_cursor::open over the catalog of @p indexes. */
    status_t open(indexes_t const& indexes, std::span<key_t const> starts) noexcept {
        return walk_.open(indexes.catalog_(), starts);
    }

    /** @ref nvme_parts_cursor::pull over the catalog of @p indexes. */
    expected<std::size_t> pull(indexes_t const& indexes, std::span<key_t> keys, links_keys_t* links_keys,
                               links_offsets_t* links_offsets, executor_t& executor) noexcept {
        return walk_.pull(indexes.catalog_(), keys, links_keys, links_offsets, executor);
    }

  private:
    walk_t walk_;
};

} // namespace unum::usearch::basic
