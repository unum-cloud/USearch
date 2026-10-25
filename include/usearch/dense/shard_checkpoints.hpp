/**
 *  @file include/usearch/dense/shard_checkpoints.hpp
 *  @author Ash Vardanian
 *  @date September 5, 2026
 *  @brief The checkpoint plans every dense tower opens over its shard set: a lone live shard's
 *      SafeGraphs monolith, or one per live shard beside a manifest, saved and restored a step at
 *      a time.
 *
 *  Both plans are components a tower returns as its own @c save_plan_t and @c load_plan_t. The
 *  live shard count picks the layout, and a load opens one shard per part, so compaction is the
 *  only thing that merges graphs. A buffer holding several parts leads with a manifest listing
 *  each by its offset inside the buffer. What a residency does differently rides in as a
 *  residency object the plan owns, answering:
 *  - `open_shard_save(shards, shard, columns, synchronization)`, one shard's own save;
 *  - `open_shard_for_part(shards, part)`, the shard a load's part lands on, placed or opened;
 *  - `open_shard_load(shards, shard, rows)`, one shard's own load at its watermark;
 *  - `land_keys(shards, shard, rows, load_plan)`, once a load's keys reached the host column;
 *  - `end_save(shards)`, once a save finished or died.
 *
 *  Checkpoints are written positionally with every header last, so a failed save reads as absent.
 */
#pragma once

#include <cstring> // `std::memset` - a buffer's head before its commit

#include <bit>         // `std::has_single_bit` - a part's connectivity
#include <span>        // `std::span` - the custom fields & the part offsets
#include <string_view> // `std::string_view` - the checkpoint paths
#include <utility>     // `std::move`, `std::exchange`

#include <numkong/capabilities.h> // `nk_kernel_name` - the metric every checkpoint names

#include "usearch/utilities/types.hpp"
#include "usearch/utilities/blocking_files.hpp"    // `mapped_file_t`, `copy_bytes` - the checkpoints' bytes
#include "usearch/utilities/ram_buffers.hpp"       // `growing_buffer` - the plans' catalogs
#include "usearch/utilities/ram_device.hpp"        // `forkunion_allocator` - the plans' placement
#include "usearch/utilities/ram_shards.hpp"        // `host_keys` - a shard's host key column
#include "usearch/utilities/safegraphs_format.hpp" // SafeGraphs monoliths, manifests & catalogs
#include "usearch/dense/types.hpp"

namespace unum::usearch::dense {

#pragma region Columns

/** The value column's dtype: the stored scalar's own token, so a packed word's suffix carries the
 *  packing and the column's shape counts logical dimensions. */
template <typename shards_type_>
constexpr dtype_t shard_value_dtype() noexcept {
    return dtype_of(shards_type_::index_t::scalar_t::dtype());
}

/** The norms column's dtype: the dot product's own token where the shards keep norms, and
 *  @c unknown_k where they keep none, so no column is written and none is read. */
template <typename shards_type_>
dtype_t shard_norms_dtype(shards_type_ const& shards) noexcept {
    using dot_result_t = typename shards_type_::index_t::scalar_t::dot_result_t;
    return !shards.empty() && shards[0].index->stores_norms() ? dtype_of(dot_result_t::dtype()) : dtype_t::unknown_k;
}

/** The shape of a norms column's row: a node's sum, then its sum of squares, both in the dot
 *  product's dtype. */
inline constexpr column_shape_t norms_column_shape_k = vector_shape_t {2};

/** The name the norms column carries among a monolith's producer columns. */
inline constexpr std::string_view norms_column_name_k = "norms";

/** The norms column's bytes in @p monolith when they are the ones a reader keeping @p dtype norms
 *  takes: a sum and a sum of squares for every node; empty otherwise, and always for an
 *  @c unknown_k @p dtype, which keeps none. */
template <typename bytes_type_>
bytes_type_ norms_column_of(safegraphs_header_t const& header, bytes_type_ monolith, dtype_t dtype) noexcept {
    safegraphs_column_t const* const norms = header.extra_column(norms_column_name_k);
    if (dtype == dtype_t::unknown_k || !norms || norms->dtype != dtype || norms->offsets ||
        norms->shape != norms_column_shape_k || norms->nodes() != header.nodes())
        return {};
    return header.data_of(monolith, *norms);
}

/** What a dense monolith's columns need to be laid out: the node count, each column's element
 *  type, and the two row widths. */
struct monolith_layout_t {

    /** Nodes every column describes. */
    std::size_t nodes = 0;

    /** Element type of the key column. */
    dtype_t key_dtype = dtype_t::unknown_k;

    /** Element type of the value column. */
    dtype_t value_dtype = dtype_t::unknown_k;

    /** Elements per value row - the logical dimensions, the packed families included. */
    std::size_t value_columns = 0;

    /** Element type of the neighbors column. */
    dtype_t neighbors_dtype = dtype_t::unknown_k;

    /** Neighbor IDs per node's list. */
    std::size_t connectivity = 0;

    /** Element type of the norms column; @c unknown_k lays out none. */
    dtype_t norms_dtype = dtype_t::unknown_k;
};

/**
 *  @brief A header of three regular columns at aligned starts - keys, fixed-width values, and
 *      fixed-degree neighbors - then the norms where @p layout names their dtype, with the data
 *      offset they follow.
 *
 *  @param[out] norms The norms column, named, which the header's producer columns view; it must
 *      outlive the header, and stays unnamed when @p layout keeps no norms.
 */
template <typename allocator_type_ = std::allocator<byte_t>>
expected<safegraphs_header_t> monolith_header(monolith_layout_t const& layout, std::size_t alignment,
                                              std::span<safegraphs_custom_field_t const> custom_fields,
                                              safegraphs_named_column_t& norms,
                                              allocator_type_ allocator = {}) noexcept {
    safegraphs_header_t header;
    header.alignment = alignment;
    header.key.dtype = layout.key_dtype;
    header.key.data = {0, layout.nodes * dtype_bytes(layout.key_dtype)};
    header.value.dtype = layout.value_dtype;
    header.value.shape = vector_shape_t {layout.value_columns};
    header.value.data.first = round_up_to_multiple(header.key.data.bound, alignment);
    header.value.data.bound = header.value.data.first + layout.nodes * header.value.row_bytes();
    header.neighbors.dtype = layout.neighbors_dtype;
    header.neighbors.shape = vector_shape_t {layout.connectivity};
    header.neighbors.data.first = round_up_to_multiple(header.value.data.bound, alignment);
    header.neighbors.data.bound = header.neighbors.data.first + layout.nodes * header.neighbors.row_bytes();
    norms = safegraphs_named_column_t {};
    if (layout.norms_dtype != dtype_t::unknown_k) {
        norms.name = safegraphs_name_t {norms_column_name_k};
        norms.column.dtype = layout.norms_dtype;
        norms.column.shape = norms_column_shape_k;
        norms.column.data.first = round_up_to_multiple(header.neighbors.data.bound, alignment);
        norms.column.data.bound = norms.column.data.first + layout.nodes * norms.column.row_bytes();
        header.extra_columns = {&norms, 1};
    }
    auto const [data_offset, status] = safegraphs_data_offset(header, custom_fields, allocator);
    header.data_offset = data_offset;
    return {header, status};
}

/** The custom fields every monolith carries: the metric and the caller's provenance; the value
 *  column's own dtype names the stored scalar. */
template <typename shards_type_>
status_t shard_custom_fields(shards_type_ const& shards, save_config_t const& config,
                             growing_buffer<safegraphs_custom_field_t, forkunion_allocator_t>& fields) noexcept {
    if (failed(fields.reserve_exactly(1 + config.provenance.size()))) return status_t::bad_alloc_k;
    fields.push_back_reserved({"metric", nk_kernel_name(shards[0].index->kernel())});
    for (safegraphs_custom_field_t const& field : config.provenance) fields.push_back_reserved(field);
    return status_t::success_k;
}

/** Shards past the retired ones: the parts a save writes. */
template <typename shards_type_>
std::size_t live_shards_(shards_type_ const& shards) noexcept {
    return shards.size() - shards.retired_shards();
}

/** The monolith header for @p nodes rows of one shard of @p shards under @p config 's alignment,
 *  its norms column laid into @p norms. */
template <typename shards_type_>
expected<safegraphs_header_t> shard_monolith_header(shards_type_ const& shards, std::size_t nodes,
                                                    save_config_t const& config,
                                                    std::span<safegraphs_custom_field_t const> fields,
                                                    safegraphs_named_column_t& norms) noexcept {
    monolith_layout_t const layout {nodes,
                                    safegraphs_key_dtype<typename shards_type_::key_t>(),
                                    shard_value_dtype<shards_type_>(),
                                    shards[0].index->dimensions(),
                                    dtype_t::u32_k,
                                    shards[0].index->connectivity(),
                                    shard_norms_dtype(shards)};
    return monolith_header(layout, config.alignment, fields, norms);
}

/** Bytes a buffer save of @p shards under @p config writes: a lone live shard's monolith, or the
 *  manifest and every live shard's part it lists; zero for an invalid config or an unplaced set.
 *  The row counts are read now, so a buffer sized here fits a save no row lands before. */
template <typename shards_type_>
expected<std::size_t> shard_storage_requirements(shards_type_ const& shards, save_config_t const& config) noexcept {
    if (shards.empty() || !shards[0].index->dimensions()) return {0, status_t::uninitialized_k};
    if (status_t const valid = config.validate(); failed(valid)) return {0, valid};
    growing_buffer<safegraphs_custom_field_t, forkunion_allocator_t> fields {shards.allocator()};
    if (status_t const named = shard_custom_fields(shards, config, fields); failed(named)) return {0, named};
    std::span<safegraphs_custom_field_t const> const fields_span = to_span(fields);

    // A header that cannot be formatted is an allocation failure the caller must see, never a zero
    // it would size a buffer by.
    status_t framing = status_t::success_k;
    std::size_t const retired = shards.retired_shards(), parts = live_shards_(shards);
    auto part_at = [&](std::size_t part) noexcept {
        std::size_t const nodes = shards[retired + part].index->size();
        safegraphs_named_column_t norms;
        auto const [header, status] = shard_monolith_header(shards, nodes, config, fields_span, norms);
        if (failed(status)) framing = status;
        return safegraphs_part_t {{}, failed(status) ? 0 : header.monolith_bytes(), nodes};
    };
    if (parts == 1) {
        std::size_t const monolith = part_at(0).bytes;
        return {monolith, framing};
    }
    growing_buffer<std::size_t, forkunion_allocator_t> offsets {shards.allocator()};
    if (failed(offsets.resize(parts))) return {0, status_t::bad_alloc_k};
    auto const [bytes, status] = safegraphs_buffer_layout(parts, part_at, config.alignment, to_writable_span(offsets));
    if (failed(framing)) return {0, framing};
    return {bytes, status};
}

#pragma endregion Columns

#pragma region Saves

/**
 *  @brief One save in flight across every live shard of a set: one shard plan and one part each,
 *      and the tail @ref finish commits - the keys, the headers, and the manifest.
 *
 *  Steps are flat and each names its shard, so a caller walks them from one thread or many;
 *  @ref steps_of names one shard's own run, which a home-first drain follows. A dying plan drops
 *  its mappings and its shard plans, leaving parts no header commits, and ends the save on its
 *  residency.
 */
template <typename shards_type_, typename residency_type_>
struct [[nodiscard]] shard_save_plan {
    using index_t = typename shards_type_::index_t;
    using key_t = typename shards_type_::key_t;

    /** One step of one shard: its shard, and that shard's own step. */
    struct step_t {

        /** The shard the rows come from. */
        std::size_t shard = 0;

        /** The shard plan's own step. */
        typename index_t::save_plan_t::step_t inner;
    };

    shard_save_plan() noexcept : shard_save_plan(forkunion_allocator_t {}) {}

    /** An empty plan over @p allocator; only @ref open ever fills it. */
    explicit shard_save_plan(forkunion_allocator_t allocator) noexcept
        : plans_(allocator), headers_(allocator), norms_columns_(allocator), mappings_(allocator), offsets_(allocator),
          fields_(allocator), path_(allocator) {}
    shard_save_plan(shard_save_plan&& other) noexcept
        : buffer_(other.buffer_), plans_(std::move(other.plans_)), headers_(std::move(other.headers_)),
          norms_columns_(std::move(other.norms_columns_)), mappings_(std::move(other.mappings_)),
          offsets_(std::move(other.offsets_)), fields_(std::move(other.fields_)), path_(std::move(other.path_)),
          residency_(std::move(other.residency_)), shards_(std::exchange(other.shards_, nullptr)) {}
    shard_save_plan& operator=(shard_save_plan&&) = delete;
    shard_save_plan(shard_save_plan const&) = delete;
    shard_save_plan& operator=(shard_save_plan const&) = delete;

    /** The mappings and the shard plans die with the plan; an unfinished one ends the save once
     *  its shard plans are gone. */
    ~shard_save_plan() noexcept {
        if (!shards_) return;
        plans_.clear();
        residency_.end_save(*shards_);
    }

    /**
     *  @brief Opens a save of @p shards under @p path: a lone live shard's monolith at the path
     *      itself, or one monolith per live shard beside the manifest @p path names, each part's
     *      IDs its shard's own.
     *
     *  Every part is framed, sized and mapped here, and every header lands in @c finish, so a
     *  failed or cancelled save reads as absent. The plan takes @p residency first, so even a
     *  refused opening ends the save there.
     */
    static expected<shard_save_plan> open(shards_type_& shards, residency_type_ residency, std::string_view path,
                                          save_config_t const& config) noexcept {
        shard_save_plan plan {shards.allocator()};
        plan.residency_ = std::move(residency);
        plan.shards_ = &shards;
        if (status_t const ready = saveable_(shards, config); failed(ready)) return {{}, ready};
        if (status_t const kept = safegraphs_keep_path(plan.path_, path); failed(kept)) return {{}, kept};
        if (status_t const opened = plan.open_files_(config); failed(opened)) return {{}, opened};
        return {std::move(plan), status_t::success_k};
    }

    /** Opens a save of @p shards into @p buffer - the in-memory sibling of the path form: a lone
     *  live shard's monolith, or the manifest leading the buffer and one part per live shard at
     *  the offsets it lists. Size the bytes by @ref shard_storage_requirements; fewer answer
     *  @c overflow_k. */
    static expected<shard_save_plan> open(shards_type_& shards, residency_type_ residency, bytes_span_t buffer,
                                          save_config_t const& config) noexcept {
        shard_save_plan plan {shards.allocator()};
        plan.residency_ = std::move(residency);
        plan.shards_ = &shards;
        if (status_t const ready = saveable_(shards, config); failed(ready)) return {{}, ready};
        if (status_t const opened = plan.open_buffer_(buffer, config); failed(opened)) return {{}, opened};
        return {std::move(plan), status_t::success_k};
    }

    /** Step @p position of the plan; an empty one past the last. */
    step_t operator[](std::size_t position) const noexcept {
        step_t step;
        if (position >= steps().size()) return step;
        std::size_t part = 0;
        std::size_t first = 0;
        for (; first + plans_[part].steps().size() <= position; ++part) first += plans_[part].steps().size();
        step.shard = shards_->retired_shards() + part;
        step.inner = plans_[part][position - first];
        return step;
    }

    /** The steps the save spans - every live shard's own, in placement order; none once spent. */
    indexed_range<shard_save_plan> steps() const noexcept {
        std::size_t count = 0;
        if (shards_)
            for (typename index_t::save_plan_t const& plan : plans_) count += plan.steps().size();
        return {this, count};
    }

    /** The contiguous run of steps shard @p shard owns - what a home-first drain takes. */
    iota_view<std::size_t> steps_of(std::size_t shard) const noexcept {
        if (!shards_ || shard < shards_->retired_shards() || shard - shards_->retired_shards() >= plans_.size())
            return {};
        std::size_t const part = shard - shards_->retired_shards();
        std::size_t first = 0;
        for (std::size_t earlier = 0; earlier != part; ++earlier) first += plans_[earlier].steps().size();
        return {first, first + plans_[part].steps().size()};
    }

    /** Lands step @p step through its own shard's plan, using @p scratch on the host residency; a
     *  device shard leaves it untouched. */
    status_t save(step_t const& step, nk::matrix_span<node_id_t> scratch) noexcept {
        if (!shards_) return status_t::uninitialized_k;
        std::size_t const part = step.shard - shards_->retired_shards();
        if (step.shard < shards_->retired_shards() || part >= plans_.size()) return status_t::config_mismatch_k;
        return plans_[part].save(step.inner, scratch);
    }

    /**
     *  @brief Pronounces the checkpoint complete: every shard plan closes, its keys and its header
     *      land in its part, and a save of several parts commits with its manifest - the commit
     *      record, so a failed save leaves parts and no checkpoint.
     *
     *  @param[in] on_saved Fires per finished shard with its index; answering @c cancel_k stops the
     *      remaining shards and reports @c cancelled_k.
     */
    template <typename progress_type_ = dummy_progress_t>
    status_t finish(progress_type_&& on_saved = {}) noexcept {
        if (!shards_) return status_t::uninitialized_k;
        std::span<safegraphs_custom_field_t const> const fields = to_span(fields_);
        std::size_t const retired = shards_->retired_shards();
        for (std::size_t part = 0; part != plans_.size(); ++part) {
            if (status_t const finished = plans_[part].finish(); failed(finished)) return finished;
            safegraphs_header_t const& header = headers_[part];
            bytes_span_t const target = target_(part);
            std::size_t const nodes = plans_[part].visible;
            // The key column is chunked, so its rows land run by run.
            bytes_span_t const key_rows = header.rows_of(target, header.key, 0, nodes);
            std::size_t written = 0;
            status_t wrote = status_t::success_k;
            host_keys((*shards_)[retired + part].keys)
                .for_each_span({0, nodes}, [&](std::span<key_t const> run) noexcept {
                    if (succeeded(wrote))
                        wrote = copy_bytes(key_rows.subspan(written, run.size_bytes()),
                                           bytes_of(run.data(), run.size()));
                    written += run.size_bytes();
                });
            if (failed(wrote)) return wrote;
            // A lone part's header is the commit record, so it lands only once no cancel can
            // follow.
            if (plans_.size() > 1)
                if (status_t const framed = frame_part_(part, fields); failed(framed)) return framed;
            if (on_saved(retired + part) == continuation_t::cancel_k) return status_t::cancelled_k;
        }
        if (status_t const committed = plans_.size() > 1 ? write_manifest_() : frame_part_(0, fields);
            failed(committed))
            return committed;
        residency_.end_save(*std::exchange(shards_, nullptr));
        return status_t::success_k;
    }

  private:

    /** Whether @p shards can write @p config: placed, reserved, tokenized, and legal. */
    static status_t saveable_(shards_type_ const& shards, save_config_t const& config) noexcept {
        static_assert(safegraphs_key_dtype<key_t>() != dtype_t::unknown_k, "The key type has no interchange token");
        if (shards.empty() || !shards[0].index->dimensions()) return status_t::uninitialized_k;
        if (shard_value_dtype<shards_type_>() == dtype_t::unknown_k) return status_t::unknown_kind_k;
        return config.validate();
    }

    /** Writes part @p part 's header and flushes its mapping, if it has one. */
    status_t frame_part_(std::size_t part, std::span<safegraphs_custom_field_t const> fields) noexcept {
        if (status_t const framed = safegraphs_write_header(target_(part), headers_[part], fields); failed(framed))
            return framed;
        if (mappings_.empty()) return status_t::success_k;
        return mappings_[part].sync();
    }

    /** The bytes part @p part lands in: its own mapping, or its slice of the caller's buffer. */
    bytes_span_t target_(std::size_t part) noexcept {
        if (!mappings_.empty()) return mappings_[part].span();
        return buffer_.subspan(offsets_[part], headers_[part].monolith_bytes());
    }

    /** Frames one header per live shard over the rows it holds now, read once, so a shard that
     *  grows between here and its watermark is refused there. */
    status_t frame_parts_(save_config_t const& config) noexcept {
        shards_type_ const& shards = *shards_;
        std::size_t const retired = shards.retired_shards(), parts = live_shards_(shards);
        if (status_t const named = shard_custom_fields(shards, config, fields_); failed(named)) return named;
        if (failed(plans_.reserve_exactly(parts)) || failed(headers_.reserve_exactly(parts)) ||
            failed(norms_columns_.reserve_exactly(parts)))
            return status_t::bad_alloc_k;
        for (std::size_t part = 0; part != parts; ++part) {
            norms_columns_.push_back_reserved(safegraphs_named_column_t {});
            auto const [header, status] = shard_monolith_header(shards, shards[retired + part].index->size(), config,
                                                                to_span(fields_), norms_columns_.back());
            if (failed(status)) return status;
            headers_.push_back_reserved(header);
        }
        return status_t::success_k;
    }

    /** Opens each live shard's own save plan over the columns of its part. */
    status_t open_shards_(save_config_t const& config) noexcept {
        std::size_t const retired = shards_->retired_shards();
        for (std::size_t part = 0; part != headers_.size(); ++part) {
            safegraphs_header_t const& header = headers_[part];
            bytes_span_t const target = target_(part);
            std::size_t const nodes = header.nodes();
            typename index_t::columns_t const columns {header.rows_of(target, header.value, 0, nodes),
                                                       header.rows_of(target, header.neighbors, 0, nodes),
                                                       header.rows_of(target, norms_columns_[part].column, 0, nodes)};
            auto [inner, status] = residency_.open_shard_save(*shards_, retired + part, columns,
                                                              config.synchronization);
            if (failed(status)) return status;
            plans_.push_back_reserved(std::move(inner));
        }
        return status_t::success_k;
    }

    /** Frames and maps a lone shard's monolith at the plan's path, or one part per live shard
     *  beside it, then opens the shards' plans over them. */
    status_t open_files_(save_config_t const& config) noexcept {
        if (status_t const framed = frame_parts_(config); failed(framed)) return framed;
        std::size_t const parts = headers_.size();
        if (failed(mappings_.reserve_exactly(parts))) return status_t::bad_alloc_k;
        std::string_view const path = to_string_view(path_);
        std::string_view const stem = safegraphs_stem(path);
        growing_buffer<char, forkunion_allocator_t> part_path {shards_->allocator()};
        for (std::size_t part = 0; part != parts; ++part) {
            std::string_view file_path = path;
            if (parts > 1) {
                auto const [named, path_status] = safegraphs_part_path(part_path, stem, part, parts);
                if (failed(path_status)) return path_status;
                file_path = named;
            }
            auto [mapping, mapping_status] = mapped_file_t::write(file_path, headers_[part].monolith_bytes());
            if (failed(mapping_status)) return mapping_status;
            mappings_.push_back_reserved(std::move(mapping));
        }
        return open_shards_(config);
    }

    /** Frames a lone shard's monolith at the head of @p buffer, or one part per live shard at the
     *  offsets the manifest leading it will list, then opens the shards' plans over them. */
    status_t open_buffer_(bytes_span_t buffer, save_config_t const& config) noexcept {
        if (status_t const framed = frame_parts_(config); failed(framed)) return framed;
        std::size_t const parts = headers_.size();
        if (failed(offsets_.resize(parts))) return status_t::bad_alloc_k;
        std::size_t bytes = headers_[0].monolith_bytes();
        if (parts > 1) {
            auto const entry_at = [this](std::size_t part) noexcept { return part_entry_(part); };
            auto const [laid, status] = safegraphs_buffer_layout(parts, entry_at, config.alignment,
                                                                 to_writable_span(offsets_));
            if (failed(status)) return status;
            bytes = laid;
        }
        if (buffer.size() < bytes) return status_t::overflow_k;
        // Whatever the caller's bytes held must not read as a checkpoint until the commit lands.
        std::memset(buffer.data(), 0, parts > 1 ? offsets_[0] : sizeof(safegraphs_prefix_t));
        buffer_ = buffer;
        return open_shards_(config);
    }

    /** The manifest entry of part @p part, before the plan names its path or offset. */
    safegraphs_part_t part_entry_(std::size_t part) const noexcept {
        return {{}, headers_[part].monolith_bytes(), headers_[part].nodes()};
    }

    /** The manifest that commits a save of several parts: at the plan's path, its entries naming
     *  their parts relative to it, or at the head of the caller's buffer, naming their offsets. */
    status_t write_manifest_() noexcept {
        std::size_t const parts = plans_.size();
        if (mappings_.empty())
            return safegraphs_write_manifest(buffer_, parts, [&](std::size_t part) noexcept {
                safegraphs_part_t entry = part_entry_(part);
                entry.offset = offsets_[part];
                return entry;
            });
        std::string_view const path = to_string_view(path_);
        std::string_view const base = safegraphs_basename(safegraphs_stem(path));
        growing_buffer<char, forkunion_allocator_t> entry_path {shards_->allocator()};
        auto part_at = [&](std::size_t part) noexcept {
            safegraphs_part_t entry = part_entry_(part);
            auto const [name, status] = safegraphs_part_path(entry_path, base, part, parts);
            entry.path = succeeded(status) ? name : std::string_view {};
            return entry;
        };
        return safegraphs_write_manifest(path, parts, part_at);
    }

    /** The caller's buffer every part lands in; empty for a save to files. */
    bytes_span_t buffer_;

    /** One plan per live shard, in placement order. */
    growing_buffer<typename index_t::save_plan_t, forkunion_allocator_t> plans_;

    /** One header per part. */
    growing_buffer<safegraphs_header_t, forkunion_allocator_t> headers_;

    /** One norms column per part, reserved whole, so each header's producer columns keep viewing
     *  it. */
    growing_buffer<safegraphs_named_column_t, forkunion_allocator_t> norms_columns_;

    /** The file each part was mapped into; empty when the caller owns the bytes. */
    growing_buffer<mapped_file_t, forkunion_allocator_t> mappings_;

    /** Where each part starts inside the caller's buffer; empty for a save to files. */
    growing_buffer<std::size_t, forkunion_allocator_t> offsets_;

    /** The custom fields every header carries, outliving the plan's steps. */
    growing_buffer<safegraphs_custom_field_t, forkunion_allocator_t> fields_;

    /** The checkpoint's own path, copied so the plan outlives the caller's view. */
    growing_buffer<char, forkunion_allocator_t> path_;

    /** What the residency does beyond the shared steps, owned for the plan's whole life. */
    [[no_unique_address]] residency_type_ residency_ {};

    /** The shards being checkpointed; null in the empty plan a refused @ref open returns, and once
     *  @c finish committed. */
    shards_type_* shards_ = nullptr;
};

#pragma endregion Saves

#pragma region Loads

/**
 *  @brief One load in flight across every shard of a set: the mapped parts, one shard plan each,
 *      and the flat steps their rows arrive in - one per part's share of one chunk or batch.
 *
 *  Each part lands on a shard of its own, placed or opened for it, and the shards past the parts
 *  stay empty; @ref finish publishes every shard at once. Every shard's claim word stays closed
 *  from opening until the plan finishes or dies, so no insertion lands beside it. A dying plan
 *  releases every claim and publishes nothing.
 */
template <typename shards_type_, typename residency_type_>
struct [[nodiscard]] shard_load_plan {
    using index_t = typename shards_type_::index_t;
    using key_t = typename shards_type_::key_t;
    using key_locator_t = typename shards_type_::key_locator_t;

    /** One step of one shard: where it lands, which part it reads, and that part's own ID space
     *  alongside the shard plan's step. */
    struct step_t {

        /** The shard the rows land in. */
        std::size_t shard = 0;

        /** The part the rows come from. */
        std::size_t part = 0;

        /** The rows of that part the shard takes - the ID space they speak. */
        iota_view<node_id_t> source_range;

        /** The shard plan's own step, cut to one part's share of one chunk or batch. */
        typename index_t::load_plan_t::step_t inner;
    };

    shard_load_plan() noexcept : shard_load_plan(forkunion_allocator_t {}) {}

    /** An empty plan over @p allocator; only @ref open ever fills it. */
    explicit shard_load_plan(forkunion_allocator_t allocator) noexcept
        : mappings_(allocator), parts_(allocator), extra_columns_(allocator), plans_(allocator) {}
    shard_load_plan(shard_load_plan&& other) noexcept
        : mappings_(std::move(other.mappings_)), parts_(std::move(other.parts_)),
          extra_columns_(std::move(other.extra_columns_)), seats_(other.seats_), plans_(std::move(other.plans_)),
          residency_(std::move(other.residency_)), shards_(std::exchange(other.shards_, nullptr)) {}
    shard_load_plan& operator=(shard_load_plan&&) = delete;
    shard_load_plan(shard_load_plan const&) = delete;
    shard_load_plan& operator=(shard_load_plan const&) = delete;

    /** The mappings die with the plan; an unfinished one rolls its shard plans back under the
     *  closed claim words, then reopens them. */
    ~shard_load_plan() noexcept {
        if (!shards_) return;
        plans_.clear();
        shards_->reopen();
    }

    /**
     *  @brief Opens a load of the checkpoint at @p path into the empty @p shards: the content says
     *      monolith or manifest, and every part's header is vetted before a byte lands.
     *
     *  The reservation comes from @p config, and part @c p lands on live shard @c p, which the
     *  residency opens past the placed ones. Every shard's claim word closes first, and the plan's
     *  death reopens them.
     */
    static expected<shard_load_plan> open(shards_type_& shards, residency_type_ residency, std::string_view path,
                                          load_config_t const& config) noexcept {
        shard_load_plan plan {shards.allocator()};
        plan.residency_ = std::move(residency);
        if (status_t const gated = plan.gate_(shards); failed(gated)) return {{}, gated};
        if (status_t const listed = safegraphs_catalog(path, plan.mappings_, plan.parts_, plan.extra_columns_,
                                                       shards.allocator());
            failed(listed))
            return {{}, listed};
        if (status_t const opened = plan.open_shards_(config); failed(opened)) return {{}, opened};
        return {std::move(plan), status_t::success_k};
    }

    /** Opens a load of the checkpoint held in @p buffer: a lone monolith, or the manifest leading
     *  the buffer and the parts it lists. */
    static expected<shard_load_plan> open(shards_type_& shards, residency_type_ residency, bytes_view_t buffer,
                                          load_config_t const& config) noexcept {
        shard_load_plan plan {shards.allocator()};
        plan.residency_ = std::move(residency);
        if (status_t const gated = plan.gate_(shards); failed(gated)) return {{}, gated};
        if (status_t const listed = safegraphs_catalog(buffer, plan.parts_, plan.extra_columns_, shards.allocator());
            failed(listed))
            return {{}, listed};
        if (status_t const opened = plan.open_shards_(config); failed(opened)) return {{}, opened};
        return {std::move(plan), status_t::success_k};
    }

    /** Rows the checkpoint lands across every shard - what its headers said. */
    std::size_t rows() const noexcept {
        std::size_t landing_rows = 0;
        for (typename index_t::load_plan_t const& plan : plans_) landing_rows += plan.rows;
        return landing_rows;
    }

    /** Step @p position of the plan; an empty one past the last. */
    step_t operator[](std::size_t position) const noexcept {
        step_t found;
        std::size_t passed = 0;
        for_each_step_([&](step_t const& step) noexcept {
            if (passed++ != position) return true;
            found = step;
            return false;
        });
        return found;
    }

    /** The steps the load spans - every shard's own, in placement order; none once spent. */
    indexed_range<shard_load_plan> steps() const noexcept {
        std::size_t count = 0;
        for_each_step_([&](step_t const&) noexcept {
            ++count;
            return true;
        });
        return {this, count};
    }

    /** The contiguous run of steps shard @p shard owns - what a home-first drain takes. */
    iota_view<std::size_t> steps_of(std::size_t shard) const noexcept {
        std::size_t first = 0, bound = 0;
        for_each_step_([&](step_t const& step) noexcept {
            first += step.shard < shard;
            bound += step.shard <= shard;
            return step.shard <= shard;
        });
        return {first, bound};
    }

    /** Lands step @p step: its part's own columns, under that part's ID space, through the shard's
     *  own plan. */
    status_t load(step_t const& step) noexcept {
        if (!shards_) return status_t::uninitialized_k;
        if (step.shard >= plans_.size() || step.part >= parts_.size()) return status_t::config_mismatch_k;
        safegraphs_catalog_part_t const& part = parts_[step.part];
        typename index_t::columns_view_t const columns {
            part.header.data_of(part.monolith, part.header.value),
            part.header.data_of(part.monolith, part.header.neighbors),
            norms_column_of(part.header, part.monolith, shard_norms_dtype(*shards_))};
        return plans_[step.shard].load(step.inner, columns, step.source_range);
    }

    /**
     *  @brief Publishes the restored graph: every shard's keys land at its base and enter the
     *      locator map, then every shard plan publishes - the order an insert keeps, so a search
     *      never reaches a published row without its key.
     *
     *  @param[in] on_loaded Fires per finished shard; answering @c cancel_k reports
     *      @c cancelled_k.
     */
    template <typename progress_type_ = dummy_progress_t>
    status_t finish(progress_type_&& on_loaded = {}) noexcept {
        if (!shards_) return status_t::uninitialized_k;
        std::size_t const retired = shards_->retired_shards();
        for (std::size_t shard = 0; shard != plans_.size(); ++shard) {
            std::size_t const first = plans_[shard].base;
            if (shard >= retired && shard - retired < parts_.size()) {
                safegraphs_catalog_part_t const& part = parts_[shard - retired];
                std::size_t const rows = part.nodes, stored_bytes = part.header.key.row_bytes();
                bytes_view_t const part_keys = part.header.rows_of(part.monolith, part.header.key, 0, rows);
                if (part_keys.size() != rows * stored_bytes) return status_t::malformed_k;
                auto const key_of = [&](std::size_t row) noexcept {
                    return safegraphs_widened_key<key_t>(part_keys.data() + row * stored_bytes, part.header.key.dtype);
                };
                if (status_t const landed = land_rows_(shard, {first, first + rows}, key_of); failed(landed))
                    return landed;
            }
            // Every plan finishes, an empty one included, since finishing releases the claim.
            if (status_t const finished = plans_[shard].finish(); failed(finished)) return finished;
        }
        for (std::size_t shard = 0; shard != plans_.size(); ++shard)
            if (on_loaded(shard) == continuation_t::cancel_k) return status_t::cancelled_k;
        shards_->reset_routing(rows());
        std::exchange(shards_, nullptr)->reopen();
        return status_t::success_k;
    }

  private:

    /** Closes every claim word of @p shards for the plan through the residency, under its tower's
     *  unique side, the plan's death reopening them, and checks emptiness beneath them, where no
     *  batch can land before the load does. */
    status_t gate_(shards_type_& shards) noexcept {
        if (shards.empty()) return status_t::uninitialized_k;
        if (status_t const closed = residency_.close(shards); failed(closed)) return closed;
        shards_ = &shards;
        return shards.entries() ? status_t::config_mismatch_k : status_t::success_k;
    }

    /** Lands the keys of @p rows, appended to shard @p shard, row @c r taking @p key_of (r):
     *  written to the host column, handed to the residency, and only then entered into the
     *  locator map, so a failed landing leaves no key resolving to rows a later insert reuses. */
    template <typename key_of_type_>
    status_t land_rows_(std::size_t shard, iota_view<std::size_t> rows, key_of_type_& key_of) noexcept {
        shards_type_& shards = *shards_;
        typename shards_type_::shard_t& one = shards[shard];
        auto& keys = host_keys(one.keys);
        std::size_t written = 0;
        keys.for_each_span(rows, [&](std::span<key_t> run) noexcept {
            for (key_t& key : run) key = key_of(written++);
        });
        if (status_t const landed = residency_.land_keys(shards, shard, rows, plans_[shard]); failed(landed))
            return landed;
        for (std::size_t row = rows.first; row != rows.bound; ++row)
            shards.locators().insert(seats_, keys[row],
                                     key_locator_t {static_cast<u32_t>(shard), static_cast<node_id_t>(row)});
        return status_t::success_k;
    }

    /** Whether @p part can land on the shards: three regular columns of their own dtypes, its keys
     *  no wider than theirs, the metric it names agreeing with theirs, the geometry @p first froze,
     *  and every column describing the same rows. */
    status_t vetted_part_(safegraphs_catalog_part_t const& part, safegraphs_header_t const& first) const noexcept {
        safegraphs_header_t const& header = part.header;
        if (header.key.offsets || header.value.offsets || header.neighbors.offsets ||
            (!part.metric.empty() &&
             nk_kernel_named(part.metric.characters, part.metric.view().size()) != (*shards_)[0].index->kernel()) ||
            !safegraphs_key_widens(header.key.dtype, safegraphs_key_dtype<key_t>()) ||
            header.value.dtype != shard_value_dtype<shards_type_>() || header.neighbors.dtype != dtype_t::u32_k ||
            header.key.shape != column_shape_t {scalar_shape_t {}} || header.value.shape != first.value.shape ||
            header.neighbors.shape != first.neighbors.shape)
            return status_t::config_mismatch_k;
        if (header.key.nodes() != part.nodes || header.value.nodes() != part.nodes ||
            header.neighbors.nodes() != part.nodes)
            return status_t::malformed_k;
        return status_t::success_k;
    }

    /**
     *  @brief Vets the plan's parts, opens the shard each one lands on through the residency,
     *      reserves every shard for its own part's rows or its share of @p config 's, and opens one
     *      load plan per shard over them, covering their keys and claiming their locator seats.
     *
     *  Part @c p lands on live shard @c p whole, so the flat steps are the intersections of its
     *  rows with the chunks or batches of that shard's plan. A part past the shard bound answers
     *  @c overflow_k, as does a residency that builds no shards where the parts outnumber them.
     */
    status_t open_shards_(load_config_t const& config) noexcept {
        shards_type_& shards = *shards_;
        if (parts_.empty()) return status_t::malformed_k;
        safegraphs_header_t const& first = parts_[0].header;
        for (safegraphs_catalog_part_t const& part : parts_)
            if (status_t const vetted = vetted_part_(part, first); failed(vetted)) return vetted;
        connectivity_t const connectivity = static_cast<connectivity_t>(
            regular_width(first.neighbors.shape).value_or(0));
        std::size_t const dimensions = regular_width(first.value.shape).value_or(0);
        if (!std::has_single_bit(static_cast<u32_t>(connectivity)) || !dimensions) return status_t::malformed_k;
        std::size_t const bound = shards.shard_capacity(), retired = shards.retired_shards();
        std::size_t total = 0;
        for (safegraphs_catalog_part_t const& part : parts_) {
            if (part.nodes > bound) return status_t::overflow_k;
            total += part.nodes;
        }

        // The frozen geometry, when there is one, must match the parts'.
        if (shards.capacity() &&
            (dimensions != shards[0].index->dimensions() || connectivity != shards[0].index->connectivity()))
            return status_t::config_mismatch_k;
        if (!shards.capacity() && !total && !config.capacity) return status_t::success_k;

        // The geometry freezes on the shards present, the shards the parts open past them are built
        // like the rest, and each shard grows ahead of its own part's rows or its share of the
        // config's, under the bound.
        typename index_t::reserve_capacity_t per_shard {};
        per_shard.connectivity = connectivity;
        per_shard.dimensions = dimensions;
        per_shard.ceiling = bound;
        for (std::size_t shard = 0; shard != shards.size(); ++shard)
            if (status_t const reserved = shards[shard].index->reserve(per_shard).status; failed(reserved))
                return reserved;
        for (std::size_t part = 0; part != parts_.size(); ++part)
            if (status_t const opened = residency_.open_shard_for_part(shards, part); failed(opened)) return opened;
        std::size_t const shards_count = shards.size();
        std::size_t const share = divide_round_up(config.capacity, shards_count - retired);
        for (std::size_t shard = retired; shard != shards_count; ++shard) {
            std::size_t const part = shard - retired;
            per_shard.capacity = min_of(max_of(part < parts_.size() ? parts_[part].nodes : 0, share), bound);
            if (status_t const reserved = shards[shard].index->reserve(per_shard).status; failed(reserved))
                return reserved;
        }
        if (status_t const indexed = shards.locators().reserve(max_of(total, config.capacity)); failed(indexed))
            return indexed;

        if (failed(plans_.reserve_exactly(shards_count))) return status_t::bad_alloc_k;
        for (std::size_t shard = 0; shard != shards_count; ++shard) {
            std::size_t const part = shard - retired;
            std::size_t const rows = shard >= retired && part < parts_.size() ? parts_[part].nodes : 0;
            auto [inner, status] = residency_.open_shard_load(shards, shard, rows);
            if (failed(status)) return status;
            plans_.push_back_reserved(std::move(inner));
            if (status_t const covered = host_keys(shards[shard].keys).cover(plans_[shard].base + plans_[shard].rows);
                failed(covered))
                return covered;
        }
        auto [seats, seat_status] = shards.locators().claim(rows());
        if (failed(seat_status)) return seat_status;
        seats_ = seats;
        return status_t::success_k;
    }

    /** Visits the flat steps in order, each part's share of each chunk or batch of its shard's
     *  plan, until @p visit answers @c false; none once spent. */
    template <typename visit_type_>
    void for_each_step_(visit_type_&& visit) const noexcept {
        if (!shards_) return;
        std::size_t const retired = shards_->retired_shards();
        for (std::size_t shard = retired; shard < plans_.size() && shard - retired < parts_.size(); ++shard) {
            typename index_t::load_plan_t const& plan = plans_[shard];
            std::size_t const part = shard - retired, first_row = plan.base, bound_row = first_row + parts_[part].nodes;
            for (typename index_t::load_plan_t::step_t const batch : plan.steps()) {
                std::size_t const low = max_of<std::size_t>(batch.index_rows.first, first_row);
                std::size_t const high = min_of<std::size_t>(batch.index_rows.bound, bound_row);
                if (low >= high) continue;
                step_t step;
                step.shard = shard;
                step.part = part;
                step.source_range = {0, static_cast<node_id_t>(parts_[part].nodes)};
                step.inner.position = batch.position;
                step.inner.index_rows = {static_cast<node_id_t>(low), static_cast<node_id_t>(high)};
                step.inner.source_rows = {static_cast<node_id_t>(low - first_row),
                                          static_cast<node_id_t>(high - first_row)};
                if (!visit(step)) return;
            }
        }
    }

    /** The files the parts were mapped from; every part's rows are read out of them. */
    growing_buffer<mapped_file_t, forkunion_allocator_t> mappings_;

    /** The checkpoint's parts, vetted at opening. */
    growing_buffer<safegraphs_catalog_part_t, forkunion_allocator_t> parts_;

    /** Every part's producer columns, which the parts' headers view. */
    growing_buffer<safegraphs_named_column_t, forkunion_allocator_t> extra_columns_;

    /** The locator seats every landed key takes, claimed at opening, so @c finish never
     *  allocates. */
    typename shards_type_::locators_map_t::seats_t seats_ {};

    /** One plan per shard, in placement order. */
    growing_buffer<typename index_t::load_plan_t, forkunion_allocator_t> plans_;

    /** What the residency does beyond the shared steps, owned for the plan's whole life. */
    [[no_unique_address]] residency_type_ residency_ {};

    /** The shards being restored, their claim words closed so no batch interleaves with the
     *  restore; null in the empty plan a refused @ref open returns, and once @c finish
     *  published. */
    shards_type_* shards_ = nullptr;
};

#pragma endregion Loads

} // namespace unum::usearch::dense
