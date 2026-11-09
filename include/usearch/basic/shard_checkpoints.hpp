/**
 *  @file include/usearch/basic/shard_checkpoints.hpp
 *  @author Ash Vardanian
 *  @date October 22, 2026
 *  @brief The checkpoints every basic tower writes and reads over its shard set: a lone live
 *      shard's SafeGraphs monolith at the path itself, or one monolith per live shard beside a
 *      manifest, the live shard count picking which; a load opens one shard per part.
 *
 *  Each shard is lent to the SafeGraphs glue as a columned index through
 *  @ref unum::usearch::basic::shard_columns, which takes its nodes through its residency's own
 *  plans, so every residency shares one framing. What a residency does differently rides in as an
 *  object answering:
 *  - <tt>columns_extents(shards, shard, codec)</tt>, one sizing pass over a shard;
 *  - <tt>open_shard_save(shards, shard, columns, codec)</tt>, one shard's own save plan, copying nodes and
 *    their keys through <tt>save(range)</tt>, the codec passed along where it takes one;
 *  - <tt>open_shard_load(shards, shard, columns)</tt>, the shard's own load plan through <tt>load(range)</tt>,
 *    whose @c finish files the keys as the shard's one writer before its watermark moves;
 *  - <tt>open_shard_for_part(shards, part)</tt>, the shard a load's part goes to, placed or
 *    opened;
 *  - <tt>reserve(shards, shard, entries)</tt>, growing a shard ahead of a load;
 *  - on device residencies, <tt>open_sources(mappings)</tt> before the shards open, which opens one
 *    source per mapped part where the residency reads files itself, so its plans read the parts'
 *    bytes from the part that holds them, and records how the load reads the parts.
 *
 *  A residency whose plans copy nodes in order is driven through
 *  @ref unum::usearch::utilities::serial_executor_t. A checkpoint in one buffer leads with the
 *  manifest when several shards are live, each part at the aligned offset it lists, as the dense
 *  tier lays one out.
 */
#pragma once

#include <cstring> // `std::memset` - a buffer's lead before its manifest commits

#include <string_view> // `std::string_view` - the checkpoint paths

#include "usearch/utilities/types.hpp"
#include "usearch/utilities/blocking_files.hpp"    // `mapped_file_t` - the parts' bytes
#include "usearch/utilities/ram_buffers.hpp"       // `growing_buffer` - the part paths & the catalog
#include "usearch/utilities/ram_device.hpp"        // `forkunion_allocator_t` - the catalog's placement
#include "usearch/utilities/safegraphs_format.hpp" // SafeGraphs monoliths, manifests & catalogs
#include "usearch/basic/ram_index.hpp"             // `columns`, `columns_extents_t`

namespace unum::usearch::basic {

/** One shard of @p shards_type_ as the columned index a SafeGraphs monolith frames, for a call. */
template <typename shards_type_, typename residency_type_>
struct shard_columns {
    using index_t = typename shards_type_::index_t;
    using key_t = typename shards_type_::key_t;
    using node_id_t = typename index_t::node_id_t;
    using columns_t = columns<key_t, node_id_t, byte_t, u64_t>;
    using columns_view_t = columns<key_t const, node_id_t const, byte_t const, u64_t const>;

    shards_type_& shards;
    residency_type_& residency;
    std::size_t shard = 0;

    /** A save of the shard's nodes and keys. */
    struct save_plan_t {
        typename index_t::save_plan_t plan;

        /**
         *  Copies @p range out through the shard's own save plan, a refusal of @p codec passing
         *  through.
         *
         *  @par Statuses:
         *  - @c uninitialized_k The shard's plan is empty or spent.
         *  - @c config_mismatch_k @p range falls outside the plan's nodes.
         */
        template <typename codec_type_>
        status_t save(iota_view<node_id_t> range, codec_type_ const& codec) noexcept {
            if constexpr (requires { plan.save(range, codec); }) return plan.save(range, codec);
            else return plan.save(range);
        }

        /**
         *  Completes the shard's own save plan.
         *
         *  @par Statuses:
         *  - @c uninitialized_k The shard's plan is empty or spent.
         *  - @c config_mismatch_k The steps covered other than every node.
         */
        status_t finish() noexcept { return plan.finish(); }
    };

    /** A load of the shard's nodes, whose own plan files their keys before the nodes publish, the
     *  order an insert keeps. */
    struct load_plan_t {
        typename index_t::load_plan_t plan;
        shard_columns* view = nullptr;

        /**
         *  Decodes @p range through the shard's own load plan, a refusal of @p codec passing
         *  through.
         *
         *  @par Statuses:
         *  - @c uninitialized_k The shard's plan is empty or spent.
         *  - @c config_mismatch_k @p range falls outside the plan's nodes.
         */
        template <typename codec_type_>
        status_t load(iota_view<node_id_t> range, codec_type_ const& codec) noexcept {
            if constexpr (requires { plan.load(range, codec); }) return plan.load(range, codec);
            else return plan.load(range);
        }

        /**
         *  Publishes the shard's nodes and keys through its own plan, then routes past them.
         *
         *  @par Statuses:
         *  - @c uninitialized_k The shard's plan is empty or spent.
         *  - @c config_mismatch_k The steps covered other than every node.
         */
        status_t finish() noexcept {
            shards_type_& shards = view->shards;
            if (status_t const finished = plan.finish(); failed(finished)) return finished;
            shards.reset_routing(shards.entries());
            return status_t::success_k;
        }
    };

    std::size_t capacity() const noexcept { return shards[shard].index->capacity(); }

    /**
     *  Grows the shard to @p entries through the residency.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k On NVIDIA, the connectivity is not a power of two or differs from
     *    the frozen one.
     *  - @c overflow_k @p entries pass the shard's node IDs or its key multimap.
     */
    expected<std::size_t> reserve(std::size_t entries) noexcept { return residency.reserve(shards, shard, entries); }

    /**
     *  The counts a save of the shard needs, through the residency.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k On NVIDIA, the shard has no stream.
     *  - @c contended_k A mutator, a batch or an open plan holds the shard.
     */
    template <typename codec_type_>
    expected<columns_extents_t> columns_extents(codec_type_ const& codec) noexcept {
        return residency.columns_extents(shards, shard, codec);
    }

    /**
     *  Opens the shard's save plan into @p columns, its keys beside the nodes.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k On NVIDIA, the shard has no stream.
     *  - @c contended_k A mutator, a batch or an open plan holds the shard.
     *  - @c overflow_k A column is shorter than its count.
     *  - @c config_mismatch_k On RAM, a node-sized column is longer than its count.
     */
    template <typename codec_type_>
    expected<save_plan_t> plan_save(columns_t columns, codec_type_ const& codec) noexcept {
        auto [plan, status] = residency.open_shard_save(shards, shard, columns, codec);
        if (failed(status)) return {{}, status};
        return {save_plan_t {std::move(plan)}, status_t::success_k};
    }

    /**
     *  @brief Opens the shard's load plan, which copies the keys beside the nodes.
     *
     *  @par Statuses:
     *  - @c overflow_k The nodes pass the shard's node IDs or its key multimap.
     *  - @c config_mismatch_k The shard already publishes nodes, or on NVIDIA has no stream.
     *  - @c uninitialized_k On NVIDIA, no reserve froze the connectivity yet.
     *  - @c unsupported_k On NVIDIA, an entry past every rung of the face's kits.
     *  - @c contended_k A mutator, a batch, a load or a plan holds the shard.
     *  - @c malformed_k On RAM, a prefix-sum column or a neighbor ID is out of shape; on NVIDIA,
     *    an offsets column.
     *
     *  Takes the staged snapshot columns on RAM, or the saved ones where the residency's shards
     *  convert them on their device.
     */
    template <typename columns_type_>
        requires requires(residency_type_& residency, shards_type_& shards, columns_type_ const& columns) {
            residency.open_shard_load(shards, std::size_t {}, columns);
        }
    expected<load_plan_t> plan_load(columns_type_ const& columns) noexcept {
        auto [plan, status] = residency.open_shard_load(shards, shard, columns);
        if (failed(status)) return {{}, status};
        return {load_plan_t {std::move(plan), this}, status_t::success_k};
    }
};

/**
 *  @brief Writes the live shards of @p shards under @p path: a lone live shard's monolith at the
 *      path itself, or one monolith per live shard beside the manifest @p path names, the manifest
 *      written last as the commit record.
 *
 *  Retired shards hold no nodes and get no part, and each part keeps its shard's own IDs. A
 *  refusal of @p codec passes through.
 *
 *  @par Statuses:
 *  - @c uninitialized_k No shard is live, a shard's plan is spent, or the pool of @p executor
 *    has no workers.
 *  - @c contended_k A mutator, a batch or an open plan holds a shard, or the pool of
 *    @p executor runs another dispatch.
 *  - @c config_mismatch_k @p config does not validate, an element has no interchange dtype, a
 *    value disagrees with its shape, a segment is misaligned, or on NVIDIA a shard has no
 *    stream.
 *  - @c invalid_argument_k A part or manifest path does not parse.
 *  - @c overflow_k A device ends before a part or the manifest, or a column is shorter than
 *    the layout needs.
 *  - @c malformed_k A written part's header does not parse back.
 */
template <typename shards_type_, typename residency_type_, typename codec_type_, typename executor_type_,
          typename progress_type_ = dummy_progress_t>
status_t save_shards(shards_type_& shards, residency_type_ residency, std::string_view path, codec_type_ const& codec,
                     executor_type_& executor, save_config_t const& config, std::string_view metric,
                     progress_type_&& progress = {}) noexcept {
    using view_t = shard_columns<shards_type_, residency_type_>;
    std::size_t const retired = shards.retired_shards(), count = shards.size() - retired;
    if (!count) return status_t::uninitialized_k;
    if (count == 1) {
        view_t view {shards, residency, retired};
        return safegraphs_write_index(path, view, codec, executor, config, metric, progress);
    }

    // Each part is sized, mapped and written whole before the next, so a stopped save leaves parts
    // no manifest names.
    growing_buffer<safegraphs_part_t, forkunion_allocator_t> parts {shards.allocator()};
    growing_buffer<char, forkunion_allocator_t> part_path {shards.allocator()}, names {shards.allocator()};
    if (failed(parts.reserve_exactly(count))) return status_t::bad_alloc_k;
    std::string_view const stem = safegraphs_stem(path);
    for (std::size_t part = 0; part != count; ++part) {
        view_t view {shards, residency, retired + part};
        auto const [bytes, bytes_status] = safegraphs_storage_requirements(view, codec, config, metric);
        if (failed(bytes_status)) return bytes_status;
        auto const [file_path, path_status] = safegraphs_part_path(part_path, stem, part, count);
        if (failed(path_status)) return path_status;
        auto [mapping, map_status] = mapped_file_t::write(file_path, bytes);
        if (failed(map_status)) return map_status;
        if (status_t const saved = safegraphs_write_index(mapping.span(), view, codec, executor, config, metric,
                                                          progress);
            failed(saved))
            return saved;
        // The header is read before the flush, which releases the mapping.
        auto const [written, written_status] = safegraphs_read_index_metadata(bytes_view_t {mapping.span()});
        if (failed(written_status)) return written_status;
        if (status_t const flushed = mapping.sync(); failed(flushed)) return flushed;
        parts.push_back_reserved({{}, bytes, written.nodes});
    }
    std::string_view const base = safegraphs_basename(stem);
    return safegraphs_write_manifest(path, count, [&](std::size_t part) noexcept {
        auto const [name, status] = safegraphs_part_path(names, base, part, count);
        safegraphs_part_t entry = parts[part];
        entry.path = succeeded(status) ? name : std::string_view {};
        return entry;
    });
}

/**
 *  @brief Every live shard's part entry, its bytes and nodes, into @p parts, and their offsets
 *      in one buffer into @p offsets, as @ref safegraphs_buffer_layout lays them; answers the
 *      buffer's whole size.
 *
 *  @par Statuses:
 *  - @c contended_k A mutator, a batch or an open plan holds a shard.
 *  - @c config_mismatch_k @p config does not validate, or an element has no interchange dtype.
 */
template <typename shards_type_, typename residency_type_, typename codec_type_>
expected<std::size_t> lay_out_shard_parts_(shards_type_& shards, residency_type_& residency, codec_type_ const& codec,
                                           save_config_t const& config, std::string_view metric,
                                           growing_buffer<safegraphs_part_t, forkunion_allocator_t>& parts,
                                           growing_buffer<std::size_t, forkunion_allocator_t>& offsets) noexcept {
    using view_t = shard_columns<shards_type_, residency_type_>;
    std::size_t const retired = shards.retired_shards(), count = shards.size() - retired;
    if (failed(parts.resize(count)) || failed(offsets.resize(count))) return {0, status_t::bad_alloc_k};
    for (std::size_t part = 0; part != count; ++part) {
        view_t view {shards, residency, retired + part};
        auto const [extents, extents_status] = view.columns_extents(codec);
        if (failed(extents_status)) return {0, extents_status};
        auto const [bytes, bytes_status] = safegraphs_storage_requirements(view, codec, config, metric);
        if (failed(bytes_status)) return {0, bytes_status};
        parts[part] = {{}, bytes, extents.nodes};
    }
    return safegraphs_buffer_layout(
        count, [&](std::size_t part) noexcept { return parts[part]; }, config.alignment, to_writable_span(offsets));
}

/**
 *  @brief Bytes a buffer @ref save_shards of @p shards writes: a lone live shard's monolith, or the
 *      manifest and every live shard's part it lists.
 *
 *  The counts are read now, so a buffer sized here fits a save no entry is inserted before.
 *
 *  @par Statuses:
 *  - @c uninitialized_k No shard is live.
 *  - @c contended_k A mutator, a batch or an open plan holds a shard.
 *  - @c config_mismatch_k @p config does not validate, or an element has no interchange dtype.
 */
template <typename shards_type_, typename residency_type_, typename codec_type_>
expected<std::size_t> shard_storage_requirements(shards_type_& shards, residency_type_ residency,
                                                 codec_type_ const& codec, save_config_t const& config,
                                                 std::string_view metric) noexcept {
    if (shards.size() == shards.retired_shards()) return {0, status_t::uninitialized_k};
    growing_buffer<safegraphs_part_t, forkunion_allocator_t> parts {shards.allocator()};
    growing_buffer<std::size_t, forkunion_allocator_t> offsets {shards.allocator()};
    return lay_out_shard_parts_(shards, residency, codec, config, metric, parts, offsets);
}

/**
 *  @brief Writes the live shards of @p shards into @p buffer: a lone live shard's monolith at its
 *      head, or the manifest leading it and one monolith per live shard at the offsets it lists,
 *      the manifest written last as the commit record.
 *
 *  Size @p buffer by @ref shard_storage_requirements. Until the commit the buffer's lead is zeroed,
 *  so a stopped save never reads as a checkpoint. A refusal of @p codec passes through.
 *
 *  @par Statuses:
 *  - @c uninitialized_k No shard is live, a shard's plan is spent, or the pool of @p executor has
 *    no workers.
 *  - @c contended_k A mutator, a batch or an open plan holds a shard, or the pool of @p executor
 *    runs another dispatch.
 *  - @c config_mismatch_k @p config does not validate, an element has no interchange dtype, a
 *    value disagrees with its shape, or a segment is misaligned.
 *  - @c overflow_k @p buffer is shorter than the checkpoint.
 */
template <typename shards_type_, typename residency_type_, typename codec_type_, typename executor_type_,
          typename progress_type_ = dummy_progress_t>
status_t save_shards(shards_type_& shards, residency_type_ residency, bytes_span_t buffer, codec_type_ const& codec,
                     executor_type_& executor, save_config_t const& config, std::string_view metric,
                     progress_type_&& progress = {}) noexcept {
    using view_t = shard_columns<shards_type_, residency_type_>;
    std::size_t const retired = shards.retired_shards(), count = shards.size() - retired;
    if (!count) return status_t::uninitialized_k;
    growing_buffer<safegraphs_part_t, forkunion_allocator_t> parts {shards.allocator()};
    growing_buffer<std::size_t, forkunion_allocator_t> offsets {shards.allocator()};
    auto const [bytes, layout_status] = lay_out_shard_parts_(shards, residency, codec, config, metric, parts, offsets);
    if (failed(layout_status)) return layout_status;
    if (buffer.size() < bytes) return status_t::overflow_k;
    std::memset(buffer.data(), 0, offsets[0]);
    for (std::size_t part = 0; part != count; ++part) {
        view_t view {shards, residency, retired + part};
        if (status_t const saved = safegraphs_write_index(buffer.subspan(offsets[part], parts[part].bytes), view, codec,
                                                          executor, config, metric, progress);
            failed(saved))
            return saved;
    }
    return safegraphs_commit_buffer(
        buffer, count, [&](std::size_t part) noexcept { return parts[part]; }, to_span(offsets));
}

/** Reads the listed @p parts into live shards @c p of @p shards, each opened by the residency
 *  past the placed ones; the body of both @ref load_shards overloads. */
template <typename shards_type_, typename residency_type_, typename codec_type_, typename executor_type_,
          typename progress_type_>
status_t load_parts_(shards_type_& shards, residency_type_& residency, std::span<safegraphs_catalog_part_t const> parts,
                     codec_type_ const& codec, executor_type_& executor, load_config_t const& config,
                     progress_type_&& progress) noexcept {
    using view_t = shard_columns<shards_type_, residency_type_>;
    std::size_t const retired = shards.retired_shards();
    for (std::size_t part = 0; part != parts.size(); ++part) {
        if (status_t const opened = residency.open_shard_for_part(shards, part); failed(opened)) return opened;
        view_t view {shards, residency, retired + part};
        if (status_t const loaded = safegraphs_read_index(parts[part].monolith, view, codec, executor, config,
                                                          progress);
            failed(loaded))
            return loaded;
    }
    return status_t::success_k;
}

/**
 *  Reads the checkpoint at @p path into the placed, empty @p shards, part @c p into live shard
 *  @c p, which the residency opens past the placed ones; the shards past the parts stay empty.
 *  A refusal of @p codec passes through.
 *
 *  @par Statuses:
 *  - @c uninitialized_k No shard is placed, on NVIDIA no reserve froze the connectivity, or the
 *    pool of @p executor has no workers.
 *  - @c config_mismatch_k The shards hold entries, a column's dtype, shape or key width differs
 *    from what the shards and @p codec take, a segment is misaligned, or on NVIDIA a shard has
 *    no stream.
 *  - @c invalid_argument_k The checkpoint or a part path does not parse.
 *  - @c malformed_k The checkpoint lists no part, or a part, its header, columns or sums do not
 *    parse.
 *  - @c overflow_k The nodes pass a shard's node IDs or key multimap.
 *  - @c contended_k The set is sealed while a shard generation must open, a mutator, batch,
 *    load or plan holds a shard, or the pool of @p executor runs another dispatch.
 *  - @c unsupported_k On NVIDIA, an entry past every rung of the face's kits.
 */
template <typename shards_type_, typename residency_type_, typename codec_type_, typename executor_type_,
          typename progress_type_ = dummy_progress_t>
status_t load_shards(shards_type_& shards, residency_type_ residency, std::string_view path, codec_type_ const& codec,
                     executor_type_& executor, load_config_t const& config, progress_type_&& progress = {}) noexcept {
    if (shards.empty()) return status_t::uninitialized_k;
    if (shards.entries()) return status_t::config_mismatch_k;
    growing_buffer<mapped_file_t, forkunion_allocator_t> mappings {shards.allocator()};
    growing_buffer<safegraphs_catalog_part_t, forkunion_allocator_t> parts {shards.allocator()};
    growing_buffer<safegraphs_named_column_t, forkunion_allocator_t> extra_columns {shards.allocator()};
    if (status_t const listed = safegraphs_catalog(path, mappings, parts, extra_columns, shards.allocator());
        failed(listed))
        return listed;
    if constexpr (requires { residency.open_sources(to_writable_span(mappings)); })
        if (status_t const opened = residency.open_sources(to_writable_span(mappings)); failed(opened)) return opened;
    return load_parts_(shards, residency, to_span(parts), codec, executor, config, progress);
}

/**
 *  @brief Reads the checkpoint @p buffer holds, a monolith or the manifest leading its parts, into
 *      the placed, empty @p shards in place, as the path overload reads a file's.
 *
 *  @par Statuses:
 *  - @c uninitialized_k No shard is placed, or the pool of @p executor has no workers.
 *  - @c config_mismatch_k The shards hold entries, a column's dtype, shape or key width differs
 *    from what the shards and @p codec take, or a segment is misaligned.
 *  - @c malformed_k The buffer lists no part, or a part, its header, columns or sums do not parse.
 *  - @c overflow_k The nodes pass a shard's node IDs or key multimap.
 *  - @c contended_k The set is sealed while a shard generation must open, a mutator, batch, load
 *    or plan holds a shard, or the pool of @p executor runs another dispatch.
 */
template <typename shards_type_, typename residency_type_, typename codec_type_, typename executor_type_,
          typename progress_type_ = dummy_progress_t>
status_t load_shards(shards_type_& shards, residency_type_ residency, bytes_view_t buffer, codec_type_ const& codec,
                     executor_type_& executor, load_config_t const& config, progress_type_&& progress = {}) noexcept {
    if (shards.empty()) return status_t::uninitialized_k;
    if (shards.entries()) return status_t::config_mismatch_k;
    growing_buffer<safegraphs_catalog_part_t, forkunion_allocator_t> parts {shards.allocator()};
    growing_buffer<safegraphs_named_column_t, forkunion_allocator_t> extra_columns {shards.allocator()};
    if (status_t const listed = checkpoint_source_t {buffer}.read_header(parts, extra_columns, no_device_copy_t {},
                                                                         shards.allocator());
        failed(listed))
        return listed;
    return load_parts_(shards, residency, to_span(parts), codec, executor, config, progress);
}

} // namespace unum::usearch::basic
