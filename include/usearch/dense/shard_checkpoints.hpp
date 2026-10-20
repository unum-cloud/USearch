/**
 *  @file include/usearch/dense/shard_checkpoints.hpp
 *  @author Ash Vardanian
 *  @date September 5, 2026
 *  @brief The checkpoint plans every dense tower opens over its shard set: one SafeGraphs monolith
 *      of every shard's rows, or one per shard beside a manifest, saved and restored a step at a
 *      time.
 *
 *  Both plans are components a tower returns as its own @c save_plan_t and @c load_plan_t. What a
 *  residency does differently rides in as a residency object the plan owns, answering:
 *  - `open_shard_save(shards, shard, columns, synchronization, first_row)`, one shard's own save;
 *  - `open_shard_load(shards, shard, rows)`, one shard's own load at its watermark;
 *  - `land_keys(shards, shard, rows, load_plan)`, once a load's keys reached the host column;
 *  - `end_save(shards)`, once a save finished or died.
 *
 *  Checkpoints are written positionally with every header last, so a failed save reads as absent.
 */
#pragma once

#include <bit>         // `std::has_single_bit` - a part's connectivity
#include <span>        // `std::span` - the custom fields & the part offsets
#include <string_view> // `std::string_view` - the checkpoint paths
#include <utility>     // `std::move`, `std::exchange`

#include "usearch/utilities/types.hpp"
#include "usearch/utilities/blocking_files.hpp"    // `mapped_file_t`, `copy_bytes` - the checkpoints' bytes
#include "usearch/utilities/ram_buffers.hpp"       // `growing_buffer` - the plans' catalogs
#include "usearch/utilities/ram_device.hpp"        // `forkunion_allocator` - the plans' placement
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

/** The custom fields every monolith carries: the metric and the caller's provenance; the value
 *  column's own dtype names the stored scalar. */
template <typename shards_type_>
status_t shard_custom_fields(shards_type_ const& shards, save_config_t const& config,
                             growing_buffer<safegraphs_custom_field_t, forkunion_allocator_t>& fields) noexcept {
    if (failed(fields.reserve_exactly(1 + config.provenance.size()))) return status_t::bad_alloc_k;
    fields.push_back_reserved({"metric", nk_kernel_name(shards.kernel())});
    for (safegraphs_custom_field_t const& field : config.provenance) fields.push_back_reserved(field);
    return status_t::success_k;
}

/** The monolith header for @p nodes rows of @p shards under @p config 's alignment. */
template <typename shards_type_>
expected<safegraphs_header_t> shard_monolith_header(shards_type_ const& shards, std::size_t nodes,
                                                    save_config_t const& config,
                                                    std::span<safegraphs_custom_field_t const> fields) noexcept {
    safegraphs_regular_shape_t const shape {nodes,
                                            safegraphs_key_dtype<typename shards_type_::key_t>(),
                                            shard_value_dtype<shards_type_>(),
                                            shards.dimensions(),
                                            dtype_t::u32_k,
                                            shards.connectivity(),
                                            shard_norms_dtype(shards)};
    return safegraphs_regular_header(shape, config.alignment, fields);
}

/** Bytes a save of @p shards under @p config writes: one monolith, or every part file and the
 *  manifest measured over a placeholder stem; zero for an invalid config or an unplaced set. The
 *  row count is read now, so a buffer sized here fits a save no row lands before. */
template <typename shards_type_>
expected<std::size_t> shard_storage_requirements(shards_type_ const& shards, save_config_t const& config) noexcept {
    if (shards.empty() || !shards.dimensions()) return {0, status_t::uninitialized_k};
    if (status_t const valid = config.validate(); failed(valid)) return {0, valid};
    growing_buffer<safegraphs_custom_field_t, forkunion_allocator_t> fields {shards.allocator()};
    if (status_t const named = shard_custom_fields(shards, config, fields); failed(named)) return {0, named};
    std::span<safegraphs_custom_field_t const> const fields_span = to_span(fields);

    // A header that cannot be formatted is an allocation failure the caller must see, never a zero
    // it would size a file by.
    status_t framing = status_t::success_k;
    auto part_bytes = [&](std::size_t nodes) noexcept {
        auto const [header, status] = shard_monolith_header(shards, nodes, config, fields_span);
        if (failed(status)) framing = status;
        return failed(status) ? 0 : header.monolith_bytes();
    };
    if (config.layout == checkpoint_layout_t::monolith_k) {
        std::size_t const monolith = part_bytes(shards.rows());
        return {monolith, framing};
    }
    std::size_t total = 0;
    for (std::size_t shard = 0; shard != shards.size(); ++shard) total += part_bytes(shards[shard].index->size());
    growing_buffer<char, forkunion_allocator_t> entry_path {shards.allocator()};
    auto part_at = [&](std::size_t part) noexcept {
        std::size_t const nodes = shards[part].index->size();
        auto const [name, status] = safegraphs_part_path(entry_path, "index", part, shards.size());
        if (failed(status)) framing = status;
        return safegraphs_part_t {name, part_bytes(nodes), nodes};
    };
    auto const [manifest, status] = safegraphs_manifest_bytes(shards.size(), part_at);
    if (failed(status)) return {0, status};
    if (failed(framing)) return {0, framing};
    return {total + manifest, status_t::success_k};
}

#pragma endregion Columns

#pragma region Saves

/**
 *  @brief One save in flight across every shard of a set: one shard plan each, the parts they write
 *      into, and the tail @ref finish commits - the keys, the headers, and the manifest.
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
        : plans_(allocator), headers_(allocator), mappings_(allocator), fields_(allocator), path_(allocator) {}
    shard_save_plan(shard_save_plan&& other) noexcept
        : layout_(other.layout_), monolith_(other.monolith_), plans_(std::move(other.plans_)),
          headers_(std::move(other.headers_)), mappings_(std::move(other.mappings_)), fields_(std::move(other.fields_)),
          path_(std::move(other.path_)), residency_(std::move(other.residency_)),
          shards_(std::exchange(other.shards_, nullptr)) {}
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
     *  @brief Opens a save of @p shards under @p path in the layout @p config names: one monolith
     *      of every shard's rows in placement order with IDs rebased, so it reads as one graph, or
     *      one monolith per shard beside a manifest.
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
        if (config.layout == checkpoint_layout_t::distributed_k) {
            if (status_t const opened = plan.open_parts_(config); failed(opened)) return {{}, opened};
            return {std::move(plan), status_t::success_k};
        }
        auto const [bytes, bytes_status] = shard_storage_requirements(shards, config);
        if (failed(bytes_status)) return {{}, bytes_status};
        auto [mapping, mapping_status] = mapped_file_t::write(path, bytes);
        if (failed(mapping_status)) return {{}, mapping_status};
        if (failed(plan.mappings_.push_back(std::move(mapping)))) return {{}, status_t::bad_alloc_k};
        if (status_t const opened = plan.open_monolith_(plan.mappings_[0].span(), config); failed(opened))
            return {{}, opened};
        return {std::move(plan), status_t::success_k};
    }

    /** Opens a save of @p shards into @p monolith as one graph - the in-memory sibling of the path
     *  form; a manifest has nowhere to resolve its parts, so @c distributed_k refuses. Size the
     *  bytes by @ref shard_storage_requirements; fewer answer @c overflow_k. */
    static expected<shard_save_plan> open(shards_type_& shards, residency_type_ residency, bytes_span_t monolith,
                                          save_config_t const& config) noexcept {
        shard_save_plan plan {shards.allocator()};
        plan.residency_ = std::move(residency);
        plan.shards_ = &shards;
        if (status_t const ready = saveable_(shards, config); failed(ready)) return {{}, ready};
        if (config.layout != checkpoint_layout_t::monolith_k) return {{}, status_t::config_mismatch_k};
        if (status_t const opened = plan.open_monolith_(monolith, config); failed(opened)) return {{}, opened};
        return {std::move(plan), status_t::success_k};
    }

    /** Step @p position of the plan; an empty one past the last. */
    step_t operator[](std::size_t position) const noexcept {
        step_t step;
        if (position >= steps().size()) return step;
        std::size_t shard = 0;
        std::size_t first = 0;
        for (; first + plans_[shard].steps().size() <= position; ++shard) first += plans_[shard].steps().size();
        step.shard = shard;
        step.inner = plans_[shard][position - first];
        return step;
    }

    /** The steps the save spans - every shard's own, in placement order; none once spent. */
    indexed_range<shard_save_plan> steps() const noexcept {
        std::size_t count = 0;
        if (shards_)
            for (typename index_t::save_plan_t const& plan : plans_) count += plan.steps().size();
        return {this, count};
    }

    /** The contiguous run of steps shard @p shard owns - what a home-first drain takes. */
    iota_view<std::size_t> steps_of(std::size_t shard) const noexcept {
        if (!shards_ || shard >= plans_.size()) return {};
        std::size_t first = 0;
        for (std::size_t earlier = 0; earlier != shard; ++earlier) first += plans_[earlier].steps().size();
        return {first, first + plans_[shard].steps().size()};
    }

    /** Lands step @p step through its own shard's plan, rebasing through @p scratch on the host
     *  residency; a device shard leaves it untouched. */
    status_t save(step_t const& step, nk::matrix_span<node_id_t> scratch) noexcept {
        if (!shards_) return status_t::uninitialized_k;
        if (step.shard >= plans_.size()) return status_t::config_mismatch_k;
        return plans_[step.shard].save(step.inner, scratch);
    }

    /**
     *  @brief Pronounces the checkpoint complete: every shard plan closes, its keys land at its
     *      base, the headers follow, and a distributed save commits with its manifest - the commit
     *      record, so a failed save leaves parts and no checkpoint.
     *
     *  @param[in] on_saved Fires per finished shard with its index; answering @c cancel_k stops the
     *      remaining shards and reports @c cancelled_k.
     */
    template <typename progress_type_ = dummy_progress_t>
    status_t finish(progress_type_&& on_saved = {}) noexcept {
        if (!shards_) return status_t::uninitialized_k;
        bool const distributed = layout_ == checkpoint_layout_t::distributed_k;
        std::span<safegraphs_custom_field_t const> const fields = to_span(fields_);
        for (std::size_t shard = 0; shard != plans_.size(); ++shard) {
            if (status_t const finished = plans_[shard].finish(); failed(finished)) return finished;
            safegraphs_header_t const& header = headers_[distributed ? shard : 0];
            bytes_span_t const target = distributed ? mappings_[shard].span() : monolith_;
            std::size_t const nodes = plans_[shard].visible;
            std::size_t const base = plans_[shard].id_shift;
            // The key column is chunked, so its rows land run by run.
            bytes_span_t const key_rows = header.rows_of(target, header.key, base, nodes);
            std::size_t written = 0;
            status_t wrote = status_t::success_k;
            (*shards_)[shard].keys.for_each_span({0, nodes}, [&](std::span<key_t const> run) noexcept {
                if (succeeded(wrote))
                    wrote = copy_bytes(key_rows.subspan(written, run.size_bytes()), bytes_of(run.data(), run.size()));
                written += run.size_bytes();
            });
            if (failed(wrote)) return wrote;
            if (distributed) {
                if (status_t const framed = safegraphs_write_header(target, header, fields); failed(framed))
                    return framed;
                if (status_t const flushed = mappings_[shard].sync(); failed(flushed)) return flushed;
            }
            if (on_saved(shard) == continuation_t::cancel_k) return status_t::cancelled_k;
        }
        if (distributed) {
            if (status_t const committed = write_manifest_(); failed(committed)) return committed;
        }
        else {
            if (status_t const framed = safegraphs_write_header(monolith_, headers_[0], fields); failed(framed))
                return framed;
            if (!mappings_.empty())
                if (status_t const flushed = mappings_[0].sync(); failed(flushed)) return flushed;
        }
        residency_.end_save(*std::exchange(shards_, nullptr));
        return status_t::success_k;
    }

  private:

    /** Whether @p shards can write @p config: placed, reserved, tokenized, and legal. */
    static status_t saveable_(shards_type_ const& shards, save_config_t const& config) noexcept {
        static_assert(safegraphs_key_dtype<key_t>() != dtype_t::unknown_k, "The key type has no interchange token");
        if (shards.empty() || !shards.dimensions()) return status_t::uninitialized_k;
        if (shard_value_dtype<shards_type_>() == dtype_t::unknown_k) return status_t::unknown_kind_k;
        return config.validate();
    }

    /** Opens shard @p shard 's own save over the @p nodes rows it writes at @p first_row of
     *  @p target; the shard plan's own @c id_shift is where its keys land in @c finish. */
    status_t open_shard_(std::size_t shard, bytes_span_t target, safegraphs_header_t const& header,
                         std::size_t first_row, std::size_t nodes, save_config_t const& config) noexcept {
        typename index_t::columns_t const columns {header.rows_of(target, header.value, first_row, nodes),
                                                   header.rows_of(target, header.neighbors, first_row, nodes),
                                                   header.rows_of(target, header.norms, first_row, nodes)};
        auto [inner, status] = residency_.open_shard_save(*shards_, shard, columns, config.synchronization, first_row);
        if (failed(status)) return status;
        plans_.push_back_reserved(std::move(inner));
        return status_t::success_k;
    }

    /** Frames @p monolith for one graph, every shard writing at its own base with its IDs rebased
     *  onto it, and opens one save plan per shard over the rows it owns. */
    status_t open_monolith_(bytes_span_t monolith, save_config_t const& config) noexcept {
        shards_type_ const& shards = *shards_;
        std::size_t const shards_count = shards.size();
        layout_ = checkpoint_layout_t::monolith_k;
        monolith_ = monolith;
        if (status_t const named = shard_custom_fields(shards, config, fields_); failed(named)) return named;
        growing_buffer<std::size_t, forkunion_allocator_t> shard_rows {shards.allocator()};
        if (failed(plans_.reserve_exactly(shards_count)) || failed(headers_.reserve_exactly(1)) ||
            failed(shard_rows.reserve_exactly(shards_count)))
            return status_t::bad_alloc_k;

        // The rows are read once, so a shard that grows between here and its watermark is refused
        // there.
        std::size_t rows = 0;
        for (std::size_t shard = 0; shard != shards_count; ++shard) {
            shard_rows.push_back_reserved(shards[shard].index->size());
            rows += shard_rows[shard];
        }
        auto const [header, status] = shard_monolith_header(shards, rows, config, to_span(fields_));
        if (failed(status)) return status;
        if (monolith.size() < header.monolith_bytes()) return status_t::overflow_k;
        headers_.push_back_reserved(header);
        for (std::size_t shard = 0, base = 0; shard != shards_count; base += shard_rows[shard], ++shard)
            if (status_t const opened = open_shard_(shard, monolith, headers_[0], base, shard_rows[shard], config);
                failed(opened))
                return opened;
        return status_t::success_k;
    }

    /** Frames, sizes and maps one part per shard beside the plan's path, and opens one save plan
     *  over each - a part's IDs stay shard-local, so its rows travel verbatim. */
    status_t open_parts_(save_config_t const& config) noexcept {
        shards_type_ const& shards = *shards_;
        std::size_t const shards_count = shards.size();
        layout_ = checkpoint_layout_t::distributed_k;
        if (status_t const named = shard_custom_fields(shards, config, fields_); failed(named)) return named;
        if (failed(plans_.reserve_exactly(shards_count)) || failed(headers_.reserve_exactly(shards_count)) ||
            failed(mappings_.reserve_exactly(shards_count)))
            return status_t::bad_alloc_k;

        std::span<safegraphs_custom_field_t const> const fields = to_span(fields_);
        std::string_view const stem = safegraphs_stem(to_string_view(path_));
        growing_buffer<char, forkunion_allocator_t> part_path {shards.allocator()};
        for (std::size_t shard = 0; shard != shards_count; ++shard) {
            std::size_t const nodes = shards[shard].index->size();
            auto const [header, header_status] = shard_monolith_header(shards, nodes, config, fields);
            if (failed(header_status)) return header_status;
            auto const [file_path, path_status] = safegraphs_part_path(part_path, stem, shard, shards_count);
            if (failed(path_status)) return path_status;
            auto [mapping, mapping_status] = mapped_file_t::write(file_path, header.monolith_bytes());
            if (failed(mapping_status)) return mapping_status;
            headers_.push_back_reserved(header);
            mappings_.push_back_reserved(std::move(mapping));
            if (status_t const opened = open_shard_(shard, mappings_[shard].span(), headers_[shard], 0, nodes, config);
                failed(opened))
                return opened;
        }
        return status_t::success_k;
    }

    /** The manifest that commits a distributed save; entries name their parts relative to it. */
    status_t write_manifest_() noexcept {
        std::size_t const shards = plans_.size();
        std::string_view const path = to_string_view(path_);
        std::string_view const base = safegraphs_basename(safegraphs_stem(path));
        growing_buffer<char, forkunion_allocator_t> entry_path {shards_->allocator()};
        auto part_at = [&](std::size_t shard) noexcept {
            auto const [name, status] = safegraphs_part_path(entry_path, base, shard, shards);
            return safegraphs_part_t {name, headers_[shard].monolith_bytes(), plans_[shard].visible};
        };
        return safegraphs_write_manifest(path, shards, part_at);
    }

    /** The layout the parts were framed in. */
    checkpoint_layout_t layout_ = checkpoint_layout_t::monolith_k;

    /** The one graph every shard writes into; empty for a distributed save. */
    bytes_span_t monolith_;

    /** One plan per shard, in placement order. */
    growing_buffer<typename index_t::save_plan_t, forkunion_allocator_t> plans_;

    /** One header per part - one in all for a monolith. */
    growing_buffer<safegraphs_header_t, forkunion_allocator_t> headers_;

    /** The files the parts were mapped into; empty when the caller owns the bytes. */
    growing_buffer<mapped_file_t, forkunion_allocator_t> mappings_;

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
 *  A shard takes its parts in order, so a plan appends them at one base and @ref finish publishes
 *  every shard at once. Every shard's claim word stays closed from opening until the plan finishes
 *  or dies, so no insertion lands beside it. A dying plan releases every claim and publishes
 *  nothing.
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
        : mappings_(allocator), parts_(allocator), landings_(allocator), plans_(allocator), steps_(allocator) {}
    shard_load_plan(shard_load_plan&& other) noexcept
        : mappings_(std::move(other.mappings_)), parts_(std::move(other.parts_)), landings_(std::move(other.landings_)),
          plans_(std::move(other.plans_)), steps_(std::move(other.steps_)), residency_(std::move(other.residency_)),
          shards_(std::exchange(other.shards_, nullptr)) {}
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
     *  The reservation comes from @p config, and the parts land per its placement - as written,
     *  part @c p on shard @c p mod S, or cut into one stripe per shard. Every shard's claim word
     *  closes first, and the plan's death reopens them.
     */
    static expected<shard_load_plan> open(shards_type_& shards, residency_type_ residency, std::string_view path,
                                          load_config_t const& config) noexcept {
        shard_load_plan plan {shards.allocator()};
        plan.residency_ = std::move(residency);
        if (status_t const gated = plan.gate_(shards); failed(gated)) return {{}, gated};
        if (status_t const listed = safegraphs_catalog(path, plan.mappings_, plan.parts_, shards.allocator());
            failed(listed))
            return {{}, listed};
        if (status_t const opened = plan.open_shards_(config); failed(opened)) return {{}, opened};
        return {std::move(plan), status_t::success_k};
    }

    /** Opens a load of one monolith held in @p monolith - the implementation the path form maps
     *  for. */
    static expected<shard_load_plan> open(shards_type_& shards, residency_type_ residency, bytes_view_t monolith,
                                          load_config_t const& config) noexcept {
        shard_load_plan plan {shards.allocator()};
        plan.residency_ = std::move(residency);
        if (status_t const gated = plan.gate_(shards); failed(gated)) return {{}, gated};
        if (status_t const listed = safegraphs_catalog_monolith(monolith, plan.parts_, shards.allocator());
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
        if (position >= steps().size()) return {};
        return steps_[position];
    }

    /** The steps the load spans - every shard's own, in placement order; none once spent. */
    indexed_range<shard_load_plan> steps() const noexcept { return {this, shards_ ? steps_.size() : 0}; }

    /** The contiguous run of steps shard @p shard owns - what a home-first drain takes. */
    iota_view<std::size_t> steps_of(std::size_t shard) const noexcept {
        if (!shards_) return {};
        std::size_t first = 0;
        while (first != steps_.size() && steps_[first].shard != shard) ++first;
        std::size_t bound = first;
        while (bound != steps_.size() && steps_[bound].shard == shard) ++bound;
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
            part.header.norms_of(part.monolith, shard_norms_dtype(*shards_))};
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
        for (std::size_t shard = 0; shard != plans_.size(); ++shard) {
            std::size_t first = plans_[shard].base;
            for (checkpoint_assignment_t const& landing : landings_) {
                std::size_t const rows = landing.rows.size();
                if (landing.shard != shard || !rows) continue;
                safegraphs_catalog_part_t const& part = parts_[landing.part];
                std::size_t const stored_bytes = part.header.key.row_bytes();
                bytes_view_t const part_keys = part.header.rows_of(part.monolith, part.header.key, landing.rows.first,
                                                                   rows);
                if (part_keys.size() != rows * stored_bytes) return status_t::malformed_k;
                auto const key_of = [&](std::size_t row) noexcept {
                    return safegraphs_widened_key<key_t>(part_keys.data() + row * stored_bytes, part.header.key.dtype);
                };
                if (status_t const landed = land_rows_(shard, {first, first + rows}, key_of); failed(landed))
                    return landed;
                first += rows;
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

    /** Closes every claim word of @p shards for the plan, whose death reopens them, and checks
     *  emptiness beneath them, where no batch can land before the load does. */
    status_t gate_(shards_type_& shards) noexcept {
        if (shards.empty()) return status_t::uninitialized_k;
        if (status_t const closed = shards.close(); failed(closed)) return closed;
        shards_ = &shards;
        return shards.rows() ? status_t::config_mismatch_k : status_t::success_k;
    }

    /** Lands the keys of @p rows, appended to shard @p shard, row @c r taking @p key_of (r):
     *  written to the host column, handed to the residency, and only then entered into the
     *  locator map, so a failed landing leaves no key resolving to rows a later insert reuses. */
    template <typename key_of_type_>
    status_t land_rows_(std::size_t shard, iota_view<std::size_t> rows, key_of_type_& key_of) noexcept {
        shards_type_& shards = *shards_;
        typename shards_type_::shard_t& one = shards[shard];
        if (status_t const covered = one.keys.cover(rows.bound); failed(covered)) return covered;
        auto [seats, seat_status] = shards.locators().claim(rows.size());
        if (failed(seat_status)) return seat_status;
        std::size_t written = 0;
        one.keys.for_each_span(rows, [&](std::span<key_t> run) noexcept {
            for (key_t& key : run) key = key_of(written++);
        });
        if (status_t const landed = residency_.land_keys(shards, shard, rows, plans_[shard]); failed(landed))
            return landed;
        for (std::size_t row = rows.first; row != rows.bound; ++row)
            shards.locators().insert(seats, one.keys[row], key_locator_t {shard, static_cast<node_id_t>(row)});
        return status_t::success_k;
    }

    /** Whether @p part can land on the shards: three regular columns of their own dtypes, its keys
     *  no wider than theirs, the metric it names agreeing with theirs, the geometry @p first froze,
     *  and every column describing the same rows. */
    status_t vetted_part_(safegraphs_catalog_part_t const& part, safegraphs_header_t const& first) const noexcept {
        safegraphs_header_t const& header = part.header;
        if (header.key.offsets || header.value.offsets || header.neighbors.offsets) return status_t::config_mismatch_k;
        if (!safegraphs_key_widens(header.key.dtype, safegraphs_key_dtype<key_t>()) ||
            header.value.dtype != shard_value_dtype<shards_type_>() || header.neighbors.dtype != dtype_t::u32_k)
            return status_t::config_mismatch_k;
        if (header.key.shape != column_shape_t {scalar_shape_t {}} || header.value.shape != first.value.shape ||
            header.neighbors.shape != first.neighbors.shape)
            return status_t::config_mismatch_k;
        if (!part.metric.empty() &&
            nk_kernel_named(part.metric.characters, part.metric.view().size()) != shards_->kernel())
            return status_t::config_mismatch_k;
        if (header.key.nodes() != part.nodes || header.value.nodes() != part.nodes ||
            header.neighbors.nodes() != part.nodes)
            return status_t::malformed_k;
        return status_t::success_k;
    }

    /**
     *  @brief Vets the plan's parts, assigns them onto the shards per @p config 's placement,
     *      reserves from it, and opens one load plan per shard over the rows it ends up with.
     *
     *  A shard takes its parts in order and appends them at one base, so the flat steps are the
     *  intersections of each part's rows with the chunks or batches they land in - a part that
     *  spans two answers two steps, and two parts sharing one answer one each.
     */
    status_t open_shards_(load_config_t const& config) noexcept {
        shards_type_& shards = *shards_;
        std::size_t const shards_count = shards.size();
        if (parts_.empty()) return status_t::malformed_k;
        safegraphs_header_t const& first = parts_[0].header;
        for (safegraphs_catalog_part_t const& part : parts_)
            if (status_t const vetted = vetted_part_(part, first); failed(vetted)) return vetted;
        connectivity_t const connectivity = static_cast<connectivity_t>(
            regular_width(first.neighbors.shape).value_or(0));
        if (!std::has_single_bit(static_cast<u32_t>(connectivity))) return status_t::malformed_k;
        std::size_t const dimensions = regular_width(first.value.shape).value_or(0);
        if (!dimensions) return status_t::malformed_k;

        // Parts onto shards per the placement, and the rows each shard ends up with.
        growing_buffer<std::size_t, forkunion_allocator_t> part_rows {shards.allocator()};
        growing_buffer<std::size_t, forkunion_allocator_t> shard_rows {shards.allocator()};
        if (failed(part_rows.reserve_exactly(parts_.size())) || failed(shard_rows.resize(shards_count)))
            return status_t::bad_alloc_k;
        for (safegraphs_catalog_part_t const& part : parts_) part_rows.push_back_reserved(part.nodes);
        status_t kept = status_t::success_k;
        assign_parts(config.placement, to_span(part_rows), shards_count,
                     [&](checkpoint_assignment_t assignment) noexcept {
                         if (failed(landings_.push_back(assignment))) kept = status_t::bad_alloc_k;
                         shard_rows[assignment.shard] += assignment.rows.size();
                     });
        if (failed(kept)) return kept;
        std::size_t widest = 0;
        for (std::size_t const rows : shard_rows) widest = max_of(widest, rows);

        // Growth ahead of the rows the parts carry - the config's or the widest shard times the
        // shard count - and the frozen geometry, when there is one, must match the parts'.
        std::size_t const wanted = max_of(config.capacity, widest * shards_count);
        if (shards.capacity() && (dimensions != shards.dimensions() || connectivity != shards.connectivity()))
            return status_t::config_mismatch_k;
        if (!shards.capacity() && !wanted) return status_t::success_k;
        if (status_t const reserved = shards.reserve({wanted, connectivity, dimensions}).status; failed(reserved))
            return reserved;

        if (failed(plans_.reserve_exactly(shards_count))) return status_t::bad_alloc_k;
        for (std::size_t shard = 0; shard != shards_count; ++shard) {
            auto [inner, status] = residency_.open_shard_load(shards, shard, shard_rows[shard]);
            if (failed(status)) return status;
            plans_.push_back_reserved(std::move(inner));
            typename index_t::load_plan_t const& opened = plans_[shard];
            std::size_t first_row = opened.base;
            for (checkpoint_assignment_t const& assignment : landings_) {
                if (assignment.shard != shard) continue;
                std::size_t const bound_row = first_row + assignment.rows.size();
                for (typename index_t::load_plan_t::step_t const batch : opened.steps()) {
                    std::size_t const low = max_of<std::size_t>(batch.index_rows.first, first_row);
                    std::size_t const high = min_of<std::size_t>(batch.index_rows.bound, bound_row);
                    if (low >= high) continue;
                    step_t step;
                    step.shard = shard;
                    step.part = assignment.part;
                    step.source_range = {static_cast<node_id_t>(assignment.rows.first),
                                         static_cast<node_id_t>(assignment.rows.bound)};
                    step.inner.position = batch.position;
                    step.inner.index_rows = {static_cast<node_id_t>(low), static_cast<node_id_t>(high)};
                    step.inner.source_rows = {static_cast<node_id_t>(assignment.rows.first + low - first_row),
                                              static_cast<node_id_t>(assignment.rows.first + high - first_row)};
                    if (failed(steps_.push_back(step))) return status_t::bad_alloc_k;
                }
                first_row = bound_row;
            }
        }
        return status_t::success_k;
    }

    /** The files the parts were mapped from; every part's rows are read out of them. */
    growing_buffer<mapped_file_t, forkunion_allocator_t> mappings_;

    /** The checkpoint's parts, vetted at opening. */
    growing_buffer<safegraphs_catalog_part_t, forkunion_allocator_t> parts_;

    /** Which part lands where, in the order each shard appends them - the sequence @c finish walks
     *  to place the keys. */
    growing_buffer<checkpoint_assignment_t, forkunion_allocator_t> landings_;

    /** One plan per shard, in placement order. */
    growing_buffer<typename index_t::load_plan_t, forkunion_allocator_t> plans_;

    /** The flat steps, grouped by shard. */
    growing_buffer<step_t, forkunion_allocator_t> steps_;

    /** What the residency does beyond the shared steps, owned for the plan's whole life. */
    [[no_unique_address]] residency_type_ residency_ {};

    /** The shards being restored, their claim words closed so no batch interleaves with the
     *  restore; null in the empty plan a refused @ref open returns, and once @c finish
     *  published. */
    shards_type_* shards_ = nullptr;
};

#pragma endregion Loads

} // namespace unum::usearch::dense
