/**
 *  @file include/usearch/basic/shard_checkpoints.hpp
 *  @author Ash Vardanian
 *  @date October 22, 2026
 *  @brief The checkpoints every basic tower writes and reads over its shard set: a lone live
 *      shard's SafeGraphs monolith at the path itself, or one monolith per live shard beside a
 *      manifest, the live shard count picking which; a load opens one shard per part.
 *
 *  Each shard is lent to the SafeGraphs glue as a columned index through @ref shard_columns, which
 *  takes its rows through its residency's own plans, so every residency shares one framing. What a
 *  residency does differently rides in as an object answering:
 *  - `columns_extents(shards, shard, codec)`, one sizing pass over a shard;
 *  - `open_shard_save(shards, shard, columns, codec)` and `open_shard_load(shards, shard,
 *    columns)`, one shard's own plans, landing rows through `save(rows)` and `load(rows)`, the
 *    codec passed along where they take one;
 *  - `save_keys(shards, shard, keys)`, the keys a save writes beside the rows;
 *  - `claim_seats(shards, rows)` and `land_keys(shards, shard, first_row, keys, seats)`, the keys a
 *    load lands beside the rows before they publish;
 *  - `open_shard_for_part(shards, part)`, the shard a load's part lands on, placed or opened;
 *  - `reserve(shards, shard, rows)`, growing a shard ahead of a load.
 *
 *  A residency whose plans land rows in order is driven through @ref serial_executor_t.
 */
#pragma once

#include <span>        // `std::span` - the keys a load lands
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

    /** A save of the shard's rows; its keys landed at planning. */
    struct save_plan_t {
        typename index_t::save_plan_t rows;

        template <typename codec_type_>
        status_t save(iota_view<node_id_t> range, codec_type_ const& codec) noexcept {
            if constexpr (requires { rows.save(range, codec); }) return rows.save(range, codec);
            else return rows.save(range);
        }
        status_t finish() noexcept { return rows.finish(); }
    };

    /** A load of the shard's rows, whose keys land through the residency before the rows publish,
     *  the order an insert keeps. */
    struct load_plan_t {
        typename index_t::load_plan_t rows;
        shard_columns* view = nullptr;
        std::span<key_t const> keys;
        typename shards_type_::locators_map_t::seats_t seats {};

        template <typename codec_type_>
        status_t load(iota_view<node_id_t> range, codec_type_ const& codec) noexcept {
            if constexpr (requires { rows.load(range, codec); }) return rows.load(range, codec);
            else return rows.load(range);
        }

        /** The row the shard's plan appends at: its own base, or zero for a plan filling an empty
         *  shard whole. */
        std::size_t first_row() const noexcept {
            if constexpr (requires { rows.base; }) return rows.base;
            else return 0;
        }

        status_t finish() noexcept {
            shards_type_& shards = view->shards;
            if (status_t const landed = view->residency.land_keys(shards, view->shard, first_row(), keys, seats);
                failed(landed))
                return landed;
            if (status_t const finished = rows.finish(); failed(finished)) return finished;
            shards.reset_routing(shards.entries());
            return status_t::success_k;
        }
    };

    std::size_t capacity() const noexcept { return shards[shard].index->capacity(); }
    expected<std::size_t> reserve(std::size_t rows) noexcept { return residency.reserve(shards, shard, rows); }

    template <typename codec_type_>
    expected<columns_extents_t> columns_extents(codec_type_ const& codec) noexcept {
        return residency.columns_extents(shards, shard, codec);
    }

    template <typename codec_type_>
    expected<save_plan_t> plan_save(columns_t columns, codec_type_ const& codec) noexcept {
        auto [rows, status] = residency.open_shard_save(shards, shard, columns, codec);
        if (failed(status)) return {{}, status};
        residency.save_keys(shards, shard, columns.keys);
        return {save_plan_t {std::move(rows)}, status_t::success_k};
    }

    expected<load_plan_t> plan_load(columns_view_t columns) noexcept {
        auto [seats, seat_status] = residency.claim_seats(shards, columns.keys.size());
        if (failed(seat_status)) return {{}, seat_status};
        auto [rows, status] = residency.open_shard_load(shards, shard, columns);
        if (failed(status)) return {{}, status};
        return {load_plan_t {std::move(rows), this, columns.keys, seats}, status_t::success_k};
    }
};

/**
 *  @brief Writes the live shards of @p shards under @p path: a lone live shard's monolith at the
 *      path itself, or one monolith per live shard beside the manifest @p path names, the manifest
 *      landing last as the commit record.
 *
 *  Retired shards hold no rows and get no part, and each part keeps its shard's own IDs.
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

/** Reads the checkpoint at @p path into the placed, empty @p shards, part @c p into live shard
 *  @c p, which the residency opens past the placed ones; the shards past the parts stay empty. */
template <typename shards_type_, typename residency_type_, typename codec_type_, typename executor_type_,
          typename progress_type_ = dummy_progress_t>
status_t load_shards(shards_type_& shards, residency_type_ residency, std::string_view path, codec_type_ const& codec,
                     executor_type_& executor, load_config_t const& config, progress_type_&& progress = {}) noexcept {
    using view_t = shard_columns<shards_type_, residency_type_>;
    if (shards.empty()) return status_t::uninitialized_k;
    if (shards.entries()) return status_t::config_mismatch_k;
    growing_buffer<mapped_file_t, forkunion_allocator_t> mappings {shards.allocator()};
    growing_buffer<safegraphs_catalog_part_t, forkunion_allocator_t> parts {shards.allocator()};
    growing_buffer<safegraphs_named_column_t, forkunion_allocator_t> extra_columns {shards.allocator()};
    if (status_t const listed = safegraphs_catalog(path, mappings, parts, extra_columns, shards.allocator());
        failed(listed))
        return listed;
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

} // namespace unum::usearch::basic
