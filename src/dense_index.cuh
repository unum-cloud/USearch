/**
 *  @file src/dense_index.cuh
 *  @author Ash Vardanian
 *  @date August 14, 2026
 *  @brief The dense modality's C surface: the handle owning one picked variant through its
 *      residency's verb table, the ladders that pick it, and every exported entry point as one
 *      visit over five residencies apiece.
 *
 *  Compiled once, as @c dense_index.cpp by the host compiler at the baseline, with
 *  @c USEARCH_TARGET_<CLASS> naming the class units the build compiled. It includes no tower: each
 *  class unit compiles its variants into residency tables, and a ladder walks the classes newest
 *  first, asking each one's table for the dtype, the serial floor last. The variants' bodies
 *  compile in the class units, never here.
 */
#pragma once

#include <cassert> // `assert` - the CPU casts never refuse
#include <cmath>   // `std::sqrt` - attention's default logit scale
#include <cstring> // `std::memcpy`, `std::memmove` - output rows and device names

#include <bit>              // `std::bit_cast` - the softmax carry crossing the ABI
#include <initializer_list> // `std::initializer_list` - the inputs a call checks against its arena
#include <memory>           // `std::unique_ptr` - the executor's topology
#include <new>              // `std::nothrow`
#include <span>             // `std::span` - key batches crossing the ABI
#include <string_view>      // `std::string_view` - checkpoint paths
#include <type_traits>      // `std::is_same_v` - the residency options' host allocator
#include <utility>          // `std::exchange` - nulling each donor slot an absorb consumes
#include <variant>          // `std::variant`, `std::visit`

#include <forkunion/forkunion.h> // `fu_topology_new` - the executor factories' discovery
#include <forkunion/atomics.hpp> // `fu::standard_atomic_ref` - the cursors' busy bits
#include <forkunion/types.hpp>   // `fu::spin_mutex` - the ticks' lock
#include <numkong/cast.h>        // `nk_cast_best` - converting foreign scalar rows at the boundary

#include "dense_punned.cuh"
#include "usearch/utilities/nvme_walks.hpp"  // `nvme_read_metadata` - an NVMe view's own header read
#include "usearch/utilities/ram_device.hpp"  // `forkunion_allocator`, `no_memory_domain_k` - the handles' host tables
#include "usearch/utilities/vram_device.hpp" // `gpu_devices`, `gpu_executor_t`, `gpu_arena_t`

#pragma region Sharded Index

/** The verb table of every residency, one of which a handle holds. */
using dense_variant_tables_t =
    std::variant<dense_verbs<vendored_residency_t::ram_k> const*, dense_verbs<vendored_residency_t::nvme_k> const*,
                 dense_verbs<vendored_residency_t::vram_nvidia_k> const*,
                 dense_verbs<vendored_residency_t::vram_amd_k> const*,
                 dense_verbs<vendored_residency_t::vram_apple_k> const*>;

/**
 *  @brief The one dynamically-typed dense index every binding consumes: the variant picked once at
 *      construction, plus the geometry metadata no shard answers.
 *
 *  Variants exist per wired @c (dtype,class) pair whose kits a class unit compiles, never the full
 *  cross product; each keeps its metric as a runtime field, since only the distance epilogue reads
 *  it. The VRAM residencies come per GPU vendor the build carries, and the read-only NVMe
 *  residency wherever a CPU class picks a kit of its own.
 */
struct usearch_dense_index_handle_t {

    /** The variant picked once at construction: its residency's verb table, and the index it
     *  serves, which only that table's entries cast back to their type. */
    dense_variant_tables_t tables;
    void* index = nullptr;

    /** Executor calls naming the handle that have not ended, which its free waits out. */
    u32_t named_calls = 0;

    usearch_dense_index_handle_t(dense_variant_tables_t picked, void* created) noexcept
        : tables(picked), index(created) {}
    usearch_dense_index_handle_t(usearch_dense_index_handle_t const&) = delete;
    usearch_dense_index_handle_t& operator=(usearch_dense_index_handle_t const&) = delete;
    ~usearch_dense_index_handle_t() noexcept {
        std::visit([&](auto const* verbs) noexcept { verbs->destroy(index); }, tables);
    }
};

/** Applies @p visitor to the variant @p handle picked, as
 *  <tt>visitor(punned_dense_index<residency>)</tt>. */
template <typename visitor_type_>
static void visit_punned(usearch_dense_index_handle_t const& handle, visitor_type_&& visitor) noexcept {
    std::visit(
        [&]<vendored_residency_t residency_>(dense_verbs<residency_> const* verbs) noexcept {
            visitor(punned_dense_index<residency_> {*verbs, handle.index});
        },
        handle.tables);
}

/** The variant of @p other when it shares the residency table of @p punned, so both are one dtype,
 *  class and residency; null otherwise. */
template <vendored_residency_t residency_>
static void* same_punned_index(punned_dense_index<residency_> const& punned,
                               usearch_dense_index_handle_t const& other) noexcept {
    dense_verbs<residency_> const* const* const verbs = std::get_if<dense_verbs<residency_> const*>(&other.tables);
    return verbs && *verbs == &punned.verbs ? other.index : nullptr;
}

#pragma endregion Sharded Index

#pragma region Construction

/** Appends one RAM shard per named memory domain - or one per allocatable domain of a transient
 *  topology when the list is empty, so a bare init still spans the machine. */
template <vendored_residency_t residency_>
static status_t place_shards(punned_dense_index<residency_> const& punned, usearch_ram_options_t const& ram) noexcept {
    punned.verbs.bound_shards(punned.index, ram.shard_capacity);
    auto const emplace = [&](fu_memory_domain_id_t domain) noexcept {
        return punned.verbs.emplace(punned.index, ram_residency_k, domain);
    };
    if (!ram.memory_domains || !ram.memory_domains_count) return emplace_on_every_domain(emplace);
    for (std::size_t position = 0; position != ram.memory_domains_count; ++position)
        if (status_t const placed = emplace(static_cast<fu_memory_domain_id_t>(ram.memory_domains[position]));
            failed(placed))
            return placed;
    return status_t::success_k;
}

/** Where an NVMe view's parts come from: the checkpoint, and one device or directory per part when
 *  any is named. */
struct nvme_placement_t {
    std::string_view path;
    std::span<std::string_view const> paths;

    /** The class the view dispatches as, @c usearch_cpu_class_unknown_k for the CPU's own. */
    usearch_cpu_class_t cpu_class = usearch_cpu_class_unknown_k;

    /** The RAM a key table may take, borrowed for the call; @c NULL holds none. */
    usearch_ram_options_t const* ram = nullptr;

    /** The executor a key table fills on; @c NULL with no key table asked for. */
    usearch_executor_t executor = nullptr;
};

/** Opens every part of the checkpoint @p nvme names, served in place, keys held when asked. */
template <vendored_residency_t residency_>
static status_t place_shards(punned_dense_index<residency_> const& punned, nvme_placement_t const& nvme) noexcept {
    return punned.verbs.view(punned.index, nvme.path, nvme.paths, nvme.ram, nvme.executor);
}

/** Appends one VRAM shard per named USearch device ordinal, on that device's vendor ordinal. */
template <vendored_residency_t residency_>
static status_t place_shards(punned_dense_index<residency_> const& punned,
                             usearch_vram_options_t const& vram) noexcept {
    punned.verbs.bound_shards(punned.index, vram.shard_capacity);
    std::span<gpu_device_t const> const devices = gpu_devices();
    for (std::size_t shard = 0; shard != named_count(vram); ++shard) {
        std::size_t const ordinal = named_ordinal(vram, shard);
        if (ordinal >= devices.size()) return status_t::device_error_k;
        if (status_t const placed = punned.verbs.emplace(punned.index, vram_residency(devices[ordinal].vendor_ordinal),
                                                         no_memory_domain_k);
            failed(placed))
            return placed;
    }
    return status_t::success_k;
}

/** Creates the index of the variant @p verbs, scoring under @p kernel, and places its shards where
 *  @p options name. The constructors below finish @p outcome only on failure, since a load still
 *  has work to do on success. */
template <vendored_residency_t residency_, typename residency_options_type_>
static usearch_dense_index_t make_dense_handle(usearch_dense_metadata_t const& geometry,
                                               residency_options_type_ const& options,
                                               dense_verbs<residency_> const& verbs, nk_kernel_kind_t kernel,
                                               usearch_outcome_t* outcome) noexcept {
    connectivity_t const connectivity = geometry.connectivity ? static_cast<connectivity_t>(geometry.connectivity)
                                                              : connectivity_32_k;
    void* const created = verbs.create(kernel, {geometry.dimensions, connectivity}, host_allocator_of(options));
    if (!created)
        return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to allocate the index"), nullptr;
    usearch_dense_index_t const handle = new (std::nothrow) usearch_dense_index_handle_t(&verbs, created);
    if (!handle)
        return verbs.destroy(created),
               propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to allocate the index"), nullptr;
    // Placement is init's job, so a bad domain fails here rather than at the first reservation.
    punned_dense_index<residency_> const punned {verbs, created};
    if (status_t const placed = place_shards(punned, options); failed(placed))
        return delete handle, propagate_error(placed, outcome), nullptr;
    return handle;
}

/** Whether @p residency_ compiled a unit for the CPU class @p cpu_class. */
template <vendored_residency_t residency_>
static constexpr bool cpu_residency_compiled(cpu_class_t cpu_class) noexcept {
    return residency_ == vendored_residency_t::nvme_k ? nvme_class_compiled(cpu_class) : cpu_class_compiled(cpu_class);
}

/** The RAM or NVMe variant of @p dtype on the newest rung of @p ladder that fits under @p ceiling,
 *  the serial floor last. A rung whose table holds no variant of the dtype picks nothing its parent
 *  does not, so the walk goes on. */
template <vendored_residency_t residency_, cpu_class_t... classes_>
static dense_verbs<residency_> const* pick_cpu_variant(cpu_ladder<classes_...>, nk_dtype_t dtype,
                                                       cpu_ceiling_t ceiling) noexcept {
    dense_verbs<residency_> const* picked = nullptr;
    auto rung = [&]<cpu_class_t class_>() noexcept {
        if constexpr (cpu_residency_compiled<residency_>(class_))
            if (!picked && cpu_rung_fits(class_, ceiling)) picked = dense_class_tables<residency_, class_>::find(dtype);
    };
    (rung.template operator()<classes_>(), ...);
    return picked;
}

/** The device variant of @p dtype on the newest rung of @p ladder the build compiled and the
 *  devices' @p available capabilities admit, whose unit alone carries kernels for them. */
template <vendored_residency_t residency_, gpu_class_t... classes_>
static dense_verbs<residency_> const* pick_gpu_variant(gpu_ladder<classes_...>, nk_dtype_t dtype,
                                                       nk_capability_t available) noexcept {
    dense_verbs<residency_> const* picked = nullptr;
    auto rung = [&]<gpu_class_t class_>() noexcept {
        if constexpr (gpu_class_compiled(class_))
            if (!picked && numkong_admits(available, class_))
                picked = dense_class_tables<residency_, class_>::find(dtype);
    };
    (rung.template operator()<classes_>(), ...);
    return picked;
}

/** Picks the RAM variant for one dtype, walking the CPU ladder newest-first; a refused rung falls
 *  to the next in the list, so lineages that share a floor interleave. */
static usearch_dense_index_t make_index(usearch_dense_metadata_t const& geometry, usearch_ram_options_t const& ram,
                                        nk_dtype_t dtype, nk_kernel_kind_t kernel,
                                        usearch_outcome_t* outcome) noexcept {
    auto const [ceiling, pick_status] = pick_cpu_ceiling(ram.cpu_class);
    if (failed(pick_status))
        return propagate_error(usearch_status_unsupported_k, outcome, "The CPU does not admit the pinned class"),
               nullptr;
    dense_verbs<vendored_residency_t::ram_k> const* const verbs = pick_cpu_variant<vendored_residency_t::ram_k>(
        cpu_ladder_t {}, dtype, ceiling);
    return make_dense_handle(geometry, ram, *verbs, kernel, outcome);
}

/** Picks the NVMe variant for one dtype on the RAM ladder's classes that compiled one. The parts'
 *  headers name the metric @p kernel was read from, so the view checks them against each other
 *  rather than against it. */
static usearch_dense_index_t make_index(usearch_dense_metadata_t const& geometry, nvme_placement_t const& nvme,
                                        nk_dtype_t dtype, nk_kernel_kind_t kernel,
                                        usearch_outcome_t* outcome) noexcept {
    auto const [ceiling, pick_status] = pick_cpu_ceiling(nvme.cpu_class);
    if (failed(pick_status))
        return propagate_error(usearch_status_unsupported_k, outcome, "The CPU does not admit the pinned class"),
               nullptr;
    dense_verbs<vendored_residency_t::nvme_k> const* const verbs = pick_cpu_variant<vendored_residency_t::nvme_k>(
        cpu_ladder_t {}, dtype, ceiling);
    return make_dense_handle(geometry, nvme, *verbs, kernel, outcome);
}

/** The class every named device shares, read off the process listing: an index spans one vendor,
 *  since its shards merge through peer copies, and one generation, since its variants run one
 *  class's kernels. Mixed vendors or generations refuse with @c config_mismatch_k, and an unlisted
 *  ordinal with @c device_error_k. */
static expected<gpu_class_t> shared_gpu_class(usearch_vram_options_t const& vram) noexcept {
    std::span<gpu_device_t const> const devices = gpu_devices();
    gpu_device_t shared;
    for (std::size_t position = 0; position != named_count(vram); ++position) {
        std::size_t const ordinal = named_ordinal(vram, position);
        if (ordinal >= devices.size()) return {gpu_class_t::unknown_k, status_t::device_error_k};
        gpu_device_t const device = devices[ordinal];
        if (position && (device.vendor != shared.vendor || gpu_class_of(device) != gpu_class_of(shared)))
            return {gpu_class_t::unknown_k, status_t::config_mismatch_k};
        shared = device;
    }
    return {gpu_class_of(shared), status_t::success_k};
}

/** The VRAM sibling of the RAM @ref make_index: the devices' vendor picks the ladder, and its
 *  classes newest-first the first rung the devices admit, narrowed to the pinned class when one is
 *  named; every dtype has a kit there, and @ref with_variant admits only the metrics its family
 *  scores by. */
static usearch_dense_index_t make_index(usearch_dense_metadata_t const& geometry, usearch_vram_options_t const& vram,
                                        nk_dtype_t dtype, nk_kernel_kind_t kernel,
                                        usearch_outcome_t* outcome) noexcept {
    auto const [gpu_class, status] = shared_gpu_class(vram);
    if (status == status_t::config_mismatch_k)
        return propagate_error(status, outcome, "One index spans one GPU vendor and shard_generation"), nullptr;
    nk_capability_t const available = gpu_pin_capabilities(gpu_class, vram.gpu_class);
    if (!available)
        return propagate_error(usearch_status_unsupported_k, outcome, "The devices do not run the pinned class"),
               nullptr;
    if (available & nk_cap_cuda_k)
        if (dense_verbs<vendored_residency_t::vram_nvidia_k> const* const verbs = pick_gpu_variant<
                vendored_residency_t::vram_nvidia_k>(
                gpu_ladder<gpu_class_t::nvidia_blackwellrtx_k, gpu_class_t::nvidia_blackwell_k,
                           gpu_class_t::nvidia_hopper_k, gpu_class_t::nvidia_ada_k, gpu_class_t::nvidia_ampere_k> {},
                dtype, available))
            return make_dense_handle(geometry, vram, *verbs, kernel, outcome);
    if (available & nk_cap_rocm_k)
        if (dense_verbs<vendored_residency_t::vram_amd_k> const* const verbs =
                pick_gpu_variant<vendored_residency_t::vram_amd_k>(
                    gpu_ladder<gpu_class_t::amd_cdna5_k, gpu_class_t::amd_cdna4_k, gpu_class_t::amd_cdna3_k> {}, dtype,
                    available))
            return make_dense_handle(geometry, vram, *verbs, kernel, outcome);
    if (available & nk_cap_metal_k)
        if (dense_verbs<vendored_residency_t::vram_apple_k> const* const verbs = pick_gpu_variant<
                vendored_residency_t::vram_apple_k>(
                gpu_ladder<gpu_class_t::apple_metal10_k, gpu_class_t::apple_metal9_k, gpu_class_t::apple_metal7_k> {},
                dtype, available))
            return make_dense_handle(geometry, vram, *verbs, kernel, outcome);
    return propagate_error(usearch_status_device_k, outcome,
                           "The devices are not listed, or this build carries no kernels for them"),
           nullptr;
}

/** Refuses at init, before any variant is picked or device touched, a geometry the engines would
 *  only refuse at the first growth: no dimensions, sub-byte rows ending inside a storage value, or
 *  a connectivity that is not a power of two within @c connectivity_t; zero derives the default. */
static bool refuses_geometry(usearch_dense_metadata_t const& geometry, usearch_outcome_t* outcome) noexcept {
    nk_dtype_t const dtype = nk_dtype_of(geometry.quantization);
    std::size_t const connectivity = geometry.connectivity;
    // A dtype the build leaves out is refused by `with_variant`, so its packing is never read here.
    bool const stored = dense_dtypes_t::index_of(dtype) != dense_dtypes_t::size_k;
    bool const filled = !stored || geometry.dimensions % nk_dimensions_per_value(dtype) == 0;
    bool const shaped = !connectivity || (std::has_single_bit(connectivity) &&
                                          connectivity == static_cast<connectivity_t>(connectivity));
    if (geometry.dimensions && filled && shaped) return false;
    return !propagate_error(usearch_status_config_mismatch_k, outcome,
                            "The dimensions or connectivity cannot shape a graph");
}

/** Routes the geometry onto its variant, built where @p options name: every numeric dtype speaks
 *  inner, angular & euclidean; the bit dtype speaks inner, hamming & jaccard. A storage dtype the
 *  build's @c USEARCH_DENSE_DTYPES left out is unsupported rather than unknown. */
template <typename residency_options_type_>
static usearch_dense_index_t with_variant(usearch_dense_metadata_t const& geometry,
                                          residency_options_type_ const& options, usearch_outcome_t* outcome) noexcept {
    nk_dtype_t const dtype = nk_dtype_of(geometry.quantization);
    if (dense_dtypes_t::index_of(dtype) == dense_dtypes_t::size_k) {
        // Only f32, f64 & the geospatial i32 cross the boundary without ever being stored as rows,
        // so any other scalar the table knows was left out.
        if (dtype == nk_dtype_unknown_k || dtype == nk_f32_k || dtype == nk_f64_k || dtype == nk_i32_k)
            return propagate_error(usearch_status_unknown_kind_k, outcome, "Unsupported quantization"), nullptr;
        return propagate_error(usearch_status_unsupported_k, outcome,
                               "This build's USEARCH_DENSE_DTYPES leaves the quantization out"),
               nullptr;
    }
    bool const bits = dtype == nk_u1_k;
    switch (geometry.metric) {
    case usearch_dense_metric_inner_k: return make_index(geometry, options, dtype, nk_kernel_dot_k, outcome);
    case usearch_dense_metric_angular_k:
        if (!bits) return make_index(geometry, options, dtype, nk_kernel_angular_k, outcome);
        break;
    case usearch_dense_metric_euclidean_k:
        if (!bits) return make_index(geometry, options, dtype, nk_kernel_euclidean_k, outcome);
        break;
    case usearch_dense_metric_hamming_k:
        if (bits) return make_index(geometry, options, dtype, nk_kernel_hamming_k, outcome);
        return propagate_error(usearch_status_unknown_kind_k, outcome, "This metric needs a bit-packed quantization"),
               nullptr;
    case usearch_dense_metric_jaccard_k:
        if (bits) return make_index(geometry, options, dtype, nk_kernel_jaccard_k, outcome);
        return propagate_error(usearch_status_unknown_kind_k, outcome, "This metric needs a bit-packed quantization"),
               nullptr;
    default: break;
    }
    return propagate_error(usearch_status_unknown_kind_k, outcome,
                           "No kernel kit is wired for this (quantization,metric) pair"),
           nullptr;
}

#pragma endregion Construction

#pragma region Multi Verbs

/** @p count rows of the C view @p rows as the engines take them, their scalar riding along. */
static us::dense::rows_view_t rows_of(usearch_rows_view_t const& rows, std::size_t count) noexcept {
    return {static_cast<byte_t const*>(rows.data), count, rows.columns, rows.stride, nk_dtype_of(rows.scalar)};
}

/** The dtype of rows in @p scalar, @p columns wide, or unknown once @p outcome refuses a
 *  scalar NumKong lacks or sub-byte rows ending inside a byte, which neither its casts nor
 *  indexes hold. */
static nk_dtype_t rows_dtype_of(usearch_scalar_t scalar, std::size_t columns, usearch_outcome_t* outcome) noexcept {
    nk_dtype_t const dtype = nk_dtype_of(scalar);
    if (dtype == nk_dtype_unknown_k) propagate_error(usearch_status_unknown_kind_k, outcome, "Unknown scalar kind");
    else if (columns % nk_dimensions_per_value(dtype))
        return propagate_error(usearch_status_shape_mismatch_k, outcome, "Sub-byte rows come in whole bytes"),
               nk_dtype_unknown_k;
    return dtype;
}

/**
 *  @brief The one loop every row verb drives on a variant that serves it: hands @p rows to @p sink
 *      chunk by chunk, each variant importing its rows into its own scalar.
 *
 *  Chunks hold whole queries of @p rows_per_query rows, one per step of @p progress, or the whole
 *  call without a handler, as each engine bounds its own scratch; progress counts queries. Each
 *  chunk is one call of the verb, so the variant rewinds its arenas once per chunk. Finishes
 *  @p outcome only on failure, leaving the success to the verb once its last output is written.
 */
template <vendored_residency_t residency_, typename sink_type_>
static bool for_each_dense_chunk(punned_dense_index<residency_> punned, usearch_executor_t executor_handle, //
                                 std::size_t rows_count, usearch_rows_view_t const& rows,                   //
                                 std::size_t rows_per_query, search_config_t config,                        //
                                 usearch_progress_t progress,                                               //
                                 usearch_traversal_stats_t* stats, usearch_outcome_t* outcome,              //
                                 sink_type_&& sink) noexcept {
    if (stats) *stats = {};
    if (!rows_count) return true;
    if (!rows.data) return propagate_error(usearch_status_invalid_argument_k, outcome, "The rows must name storage");
    nk_dtype_t const source_dtype = rows_dtype_of(rows.scalar, rows.columns, outcome);
    if (source_dtype == nk_dtype_unknown_k) return false;
    usearch_executor_handle_t call_executor;
    typename punned_dense_index<residency_>::executor_t* const executor = executor_of(punned, executor_handle,
                                                                                      call_executor, outcome);
    if (!executor) return false;
    if (rows.stride && rows.stride < us::dense::packed_bytes(source_dtype, rows.columns))
        return propagate_error(usearch_status_unsupported_layout_k, outcome, "The stride is shorter than one row");

    std::size_t const queries = rows_count / rows_per_query;
    progress_ticks_t ticks {progress, queries};
    std::size_t const chunk = rows_per_query * ticks.slice();
    us::dense::rows_view_t const all = rows_of(rows, rows_count);
    for (std::size_t resolved = 0; resolved != rows_count;) {
        std::size_t const taking = min_of(chunk, rows_count - resolved);
        expected<traversal_stats_t> const found = sink(all.slice(resolved, taking), *executor, config, resolved,
                                                       taking);
        if (!crossed(found, stats, outcome)) return false;
        resolved += taking;
        if (ticks(resolved / rows_per_query, queries) == continuation_t::cancel_k)
            return propagate_error(usearch_status_cancelled_k, outcome, "The handler stopped the call");
    }
    return true;
}

/** The matching verbs' sink on one serving variant: keys and scores go to @p outputs as the walk
 *  writes them, viewed as one row of @c wanted per output row; @p rows_per_query input rows share
 *  one output row, and @p count output rows land, each chunk taking its rows of @p hints. */
template <vendored_residency_t residency_, typename verb_type_>
static bool match_dense_rows(punned_dense_index<residency_> punned, usearch_executor_t executor, //
                             std::size_t count, usearch_rows_view_t const& rows,                 //
                             std::size_t rows_per_query, ranked_hints_view_t hints,              //
                             matches_outputs_t const& outputs,                                   //
                             search_config_t config, usearch_progress_t progress,                //
                             usearch_traversal_stats_t* stats, usearch_outcome_t* outcome,       //
                             verb_type_&& verb) noexcept {
    using key_t = typename dense_verbs<residency_>::key_t;
    static_assert(sizeof(key_t) == sizeof(usearch_key_t), "Keys land in the arena as the walk writes them");
    std::size_t const wanted = outputs.wanted;
    nk::matrix_span<key_t> const keys {reinterpret_cast<key_t*>(outputs.keys), count, wanted};
    nk::matrix_span<f64_t> const scores {outputs.scores, count, wanted};
    auto sink = [&](us::dense::rows_view_t chunk_rows, typename punned_dense_index<residency_>::executor_t& executor,
                    search_config_t chunk_config, std::size_t resolved, std::size_t taking) noexcept {
        std::size_t const first = resolved / rows_per_query, matched = taking / rows_per_query;
        nk::range_t const chunk_range = nk::range(first, first + matched);
        ranked_matches_span_t const matches {keys(chunk_range, nk::slice), scores(chunk_range, nk::slice),
                                             std::span<std::size_t> {outputs.counts + first, matched}};
        ranked_hints_view_t const chunk_hints = hints.counts.empty()
                                                    ? ranked_hints_view_t {}
                                                    : ranked_hints_view_t {hints.keys(chunk_range, nk::slice),
                                                                           hints.counts.subspan(first, matched)};
        return verb(chunk_rows, chunk_hints, matches, chunk_config, executor);
    };
    return for_each_dense_chunk(punned, executor, count * rows_per_query, rows, rows_per_query, config, progress, stats,
                                outcome, sink);
}

/** The C hints @p given of @p count queries into @p hints, none when @p given is @c NULL; false
 *  once @p outcome records why they do not parse. */
static bool hints_of_view(usearch_matches_view_t const* given, std::size_t count, ranked_hints_view_t& hints,
                          usearch_outcome_t* outcome) noexcept {
    if (rejects_hints(given, count, outcome)) return false;
    if (given && count)
        hints = {{reinterpret_cast<us::utilities::uuid128_t const*>(given->keys), count, given->columns},
                 {given->counts, count}};
    return true;
}

/** Whether every query @p query_offsets cuts holds rows; false once @p outcome records why not. */
static bool accepts_query_offsets(std::span<std::size_t const> query_offsets, usearch_outcome_t* outcome) noexcept {
    for (std::size_t query = 0; query + 1 < query_offsets.size(); ++query)
        if (query_offsets[query + 1] <= query_offsets[query])
            return propagate_error(usearch_status_shape_mismatch_k, outcome, "Every query needs at least one row");
    return true;
}

#pragma endregion Multi Verbs

/** What @c usearch_dense_cursor_t owns: the walk state of one variant, with no hold on its index,
 *  and the table whose entries step and free it, which outlives every index. */
struct usearch_dense_cursor_handle_t {
    dense_variant_tables_t tables;
    void* walk = nullptr;

    /** The handle the cursor was opened on, only ever compared with the one each pull passes. */
    usearch_dense_index_handle_t const* opened_on = nullptr;

    /** Set while a pull runs, so an overlapping one is refused rather than racing it. */
    u32_t busy = 0;

    /** Executor calls naming the cursor that have not ended, which its free waits out. */
    u32_t named_calls = 0;

    usearch_dense_cursor_handle_t(dense_variant_tables_t picked, usearch_dense_index_handle_t const* index) noexcept
        : tables(picked), opened_on(index) {}
    usearch_dense_cursor_handle_t(usearch_dense_cursor_handle_t const&) = delete;
    usearch_dense_cursor_handle_t& operator=(usearch_dense_cursor_handle_t const&) = delete;
    ~usearch_dense_cursor_handle_t() noexcept {
        if (walk) std::visit([&](auto const* verbs) noexcept { verbs->cursor_free(walk); }, tables);
    }
};

/** What @c usearch_dense_refinement_t owns: one variant's plan with the donors it absorbs, which
 *  keeps its index closed until it dies, and the caller's donor slots it nulls. */
struct usearch_dense_refinement_handle_t {
    dense_variant_tables_t tables;
    void* refinement = nullptr;

    /** The handle and executor the refinement was opened with, only ever compared with the ones
     *  each call passes. */
    usearch_dense_index_handle_t const* opened_on = nullptr;
    usearch_executor_t opened_with = nullptr;

    /** The caller's donor array, @c count slots. */
    usearch_dense_index_t* donors = nullptr;
    std::size_t count = 0;

    /** Executor calls naming the refinement that have not ended, which its free waits out. */
    u32_t named_calls = 0;

    usearch_dense_refinement_handle_t(dense_variant_tables_t picked, usearch_dense_index_handle_t const* index,
                                      usearch_executor_t executor, usearch_dense_index_t* slots,
                                      std::size_t slots_count) noexcept
        : tables(picked), opened_on(index), opened_with(executor), donors(slots), count(slots_count) {}
    usearch_dense_refinement_handle_t(usearch_dense_refinement_handle_t const&) = delete;
    usearch_dense_refinement_handle_t& operator=(usearch_dense_refinement_handle_t const&) = delete;
    ~usearch_dense_refinement_handle_t() noexcept {
        if (refinement) std::visit([&](auto const* verbs) noexcept { verbs->refinement_free(refinement); }, tables);
    }
};

/** Fills every field a checkpoint holds; the type carries nothing a checkpoint cannot say. */
static void adopt_metadata(safegraphs_metadata_t const& source, usearch_dense_metadata_t* metadata) noexcept {
    metadata->dimensions = regular_width(source.value_shape).value_or(0);
    metadata->connectivity = source.connectivity;
    metadata->quantization = usearch_scalar_of(dense::nk_dtype_of(source.value_dtype));
    std::string_view const metric = source.metric.view();
    metadata->metric = usearch_metric_of(nk_kernel_named(metric.data(), metric.size()));
}

/** Builds a handle from the geometry @p source holds, but for a non-zero @p connectivity, placed
 *  where @p options name, and copies @p source into it; a failed or cancelled load frees the handle
 *  and leaves @c *index null. */
template <typename source_type_, typename residency_options_type_>
static void load_dense_index(source_type_ source, residency_options_type_ const& options, std::size_t connectivity,
                             usearch_executor_t executor, usearch_progress_t progress, usearch_dense_index_t* index,
                             usearch_outcome_t* outcome) noexcept {
    *index = nullptr;
    constexpr bool from_path_k = std::is_same_v<source_type_, std::string_view>;
    bool const empty = [&]() noexcept {
        if constexpr (from_path_k) return source.empty();
        else return source.bytes.empty();
    }();
    if (empty)
        return propagate_error(usearch_status_invalid_argument_k, outcome, "The checkpoint must not be empty"), void();
    auto const [checkpoint, read_status] = [&]() noexcept {
        if constexpr (from_path_k) return safegraphs_read_metadata(source);
        else return safegraphs_read_metadata(source, copy_off_device);
    }();
    if (failed(read_status)) return propagate_error(read_status, outcome), void();
    usearch_dense_metadata_t geometry {};
    adopt_metadata(checkpoint, &geometry);
    // A narrower connectivity reaches the shards too, whose load refuses it.
    if (connectivity) geometry.connectivity = connectivity;
    // More parts than named devices keep a shard each, part `p` on device `p` modulo their count.
    residency_options_type_ placed = options;
    growing_buffer<int32_t> part_devices {global_allocator};
    if constexpr (std::is_same_v<residency_options_type_, usearch_vram_options_t>)
        if (checkpoint.parts > named_count(options)) {
            if (failed(part_devices.resize(checkpoint.parts)))
                return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to list the parts' devices"),
                       void();
            for (std::size_t part = 0; part != part_devices.size(); ++part)
                part_devices[part] = static_cast<int32_t>(named_ordinal(options, part));
            placed.devices = part_devices.data();
            placed.devices_count = checkpoint.parts;
        }
    usearch_dense_index_t const loaded = with_variant(geometry, placed, outcome);
    if (!loaded) return;
    bool filled = false;
    visit_punned(*loaded, [&]<vendored_residency_t residency_>(punned_dense_index<residency_> punned) noexcept {
        usearch_executor_handle_t call_executor;
        typename punned_dense_index<residency_>::executor_t* const typed = executor_of(punned, executor, call_executor,
                                                                                       outcome);
        if (!typed) return;
        progress_ticks_t ticks {progress, punned.verbs.shards_count(punned.index)};
        status_t const status = [&]() noexcept {
            if constexpr (std::is_same_v<source_type_, std::string_view>)
                return punned.verbs.load(punned.index, source, *typed, ticks);
            else return punned.verbs.load_buffer(punned.index, source, *typed, ticks);
        }();
        filled = succeeded(status) || propagate_error(status, outcome);
    });
    if (!filled) return delete loaded, void();
    *index = loaded;
    propagate_error(usearch_status_success_k, outcome, nullptr);
}

/** Whether @p punned takes writes and each of the @p count handles @p others names is of the very
 *  variant @p punned is, waiting out the calls naming them first; false once @p outcome records why
 *  not. The handle @p self appears among them only when no @p refusal is named. */
template <vendored_residency_t residency_>
static bool accepts_dense_variants(punned_dense_index<residency_> const& punned, usearch_dense_index_t const* others,
                                   std::size_t count, usearch_dense_index_t self, char const* refusal,
                                   usearch_outcome_t* outcome) noexcept {
    if (status_t const served = punned.verbs.serves_writes(punned.index); failed(served))
        return propagate_error(served, outcome);
    for (std::size_t position = 0; position != count; ++position) {
        usearch_dense_index_t const other = others[position];
        if (!other) return propagate_error(usearch_status_invalid_argument_k, outcome, "A handle must not be null");
        // Ahead of the wait below, which a deferred call would spend on its own handle.
        if (other == self && refusal) return propagate_error(usearch_status_config_mismatch_k, outcome, refusal);
        // Every scalar, class and residency is a variant of its own, so another shape is another
        // table.
        if (!same_punned_index(punned, *other))
            return propagate_error(usearch_status_config_mismatch_k, outcome,
                                   "Every handle must share the index's residency, scalar and class");
        if (other != self) wait_named_calls(other->named_calls);
    }
    return true;
}

/** Collects the indexes of the @p count handles @p others names into @p indexes, once
 *  @ref accepts_dense_variants admits them; false once @p outcome records why not. */
template <vendored_residency_t residency_>
static bool collect_variants(punned_dense_index<residency_> const& punned, usearch_dense_index_t const* others,
                             std::size_t count, usearch_dense_index_t self, char const* refusal,
                             growing_buffer<void*>& indexes, usearch_outcome_t* outcome) noexcept {
    if (!accepts_dense_variants(punned, others, count, self, refusal, outcome)) return false;
    if (failed(indexes.reserve_exactly(count))) return propagate_error(status_t::bad_alloc_k, outcome);
    for (std::size_t position = 0; position != count; ++position) indexes.push_back_reserved(others[position]->index);
    return true;
}

/** The variants behind the @p count handles @p others names, read in place. */
inline variants_view_t view_dense_variants(usearch_dense_index_t const* others, std::size_t count) noexcept {
    return {others, others ? count : 0, [](void const* handles, std::size_t position) noexcept -> void* {
                usearch_dense_index_t const* const typed = static_cast<usearch_dense_index_t const*>(handles);
                return typed[position]->index;
            }};
}

/** The tape of @p count queries, @p stride pairs of room each, carved where @p arena hands outputs
 *  out, on its device when it has one and on its host pages otherwise, its first offset zeroed
 *  there; the verbs hand every window of it the whole keys and scores with its own offsets. */
static expected<products_tape_t> carve_products_tape(usearch_arena_handle_t& arena, std::size_t count,
                                                     std::size_t stride) noexcept {
    using key_t = us::utilities::uuid128_t;
    static_assert(sizeof(key_t) == sizeof(usearch_key_t), "Keys land in the arena as the walk writes them");
    std::size_t const room = count * stride;
    products_tape_t tape;
    tape.stride = stride;
    std::size_t const base = 0;
    std::size_t* offsets = nullptr;
    key_t* keys = nullptr;
    usearch_distance_t* scores = nullptr;
    if (arena.device_memory) {
        offsets = carve<std::size_t>(arena.device_memory, count + 1);
        keys = carve<key_t>(arena.device_memory, room);
        scores = carve<usearch_distance_t>(arena.device_memory, room);
    }
    else {
        offsets = carve<std::size_t>(arena.memory, count + 1);
        keys = carve<key_t>(arena.memory, room);
        scores = carve<usearch_distance_t>(arena.memory, room);
    }
    if (!offsets || (room && (!keys || !scores))) return {tape, status_t::bad_alloc_k};
    tape.offsets = {offsets, count + 1};
    tape.keys = {keys, room};
    tape.scores = {scores, room};
    if (!arena.device_memory) {
        tape.offsets.front() = base;
        return {tape, status_t::success_k};
    }
    return {tape, drained(arena, arena.device_memory.upload(offsets, &base, sizeof(base)))};
}

/** The C tape over @p tape once every query published: the pairs @p written across the queries,
 *  no keys or scores for none. */
static usearch_matches_tape_view_t tape_view_of(products_tape_t const& tape, std::size_t written) noexcept {
    return {written ? reinterpret_cast<usearch_key_t const*>(tape.keys.data()) : nullptr,
            written ? tape.scores.data() : nullptr, tape.offsets.data()};
}

extern "C" {

#pragma region Lifetime

USEARCH_EXPORT void usearch_dense_init_ram(usearch_dense_init_ram_t* arguments) {
    call_outcome_t outcome(*arguments);
    *arguments->index = nullptr;
    usearch_dense_metadata_t const geometry {arguments->metric, arguments->quantization, arguments->dimensions,
                                             arguments->connectivity};
    if (refuses_geometry(geometry, outcome)) return;
    usearch_dense_index_t const index = with_variant(geometry, arguments->ram, outcome);
    if (!index) return;
    *arguments->index = index;
    propagate_error(usearch_status_success_k, outcome, nullptr);
}

USEARCH_EXPORT void usearch_dense_init_vram(usearch_dense_init_vram_t* arguments) {
    call_outcome_t outcome(*arguments);
    *arguments->index = nullptr;
    usearch_dense_metadata_t const geometry {arguments->metric, arguments->quantization, arguments->dimensions,
                                             arguments->connectivity};
    if (refuses_geometry(geometry, outcome)) return;
    usearch_dense_index_t const index = with_variant(geometry, arguments->vram, outcome);
    if (!index) return;
    *arguments->index = index;
    propagate_error(usearch_status_success_k, outcome, nullptr);
}

USEARCH_EXPORT void usearch_dense_view_ram(usearch_dense_view_ram_t* arguments) {
    call_outcome_t outcome(*arguments);
    *arguments->index = nullptr;
    propagate_error(usearch_status_unsupported_k, outcome,
                    "RAM shards copy a checkpoint in and cannot serve a mapping in place yet");
}

USEARCH_EXPORT void usearch_dense_view_nvme(usearch_dense_view_nvme_t* arguments) {
    if (submitted<usearch_dense_view_nvme>(arguments)) return;
    call_outcome_t outcome(*arguments);
    usearch_nvme_options_t const& nvme = arguments->nvme;
    *arguments->index = nullptr;
    if (arguments->executor && !std::holds_alternative<nvme_executor_t>(arguments->executor->executor))
        return propagate_error(usearch_status_device_k, outcome,
                               "The executor's kind must match the handle's residency"),
               void();
    if (!arguments->path.data || !arguments->path.length || (nvme.paths_count && !nvme.paths))
        return propagate_error(usearch_status_invalid_argument_k, outcome, "The paths must name storage"), void();

    std::size_t const paths_count = nvme.paths ? nvme.paths_count : 0;
    growing_buffer<std::string_view> paths {global_allocator};
    if (failed(paths.resize(paths_count)))
        return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to list the paths"), void();
    for (std::size_t path = 0; path != paths_count; ++path) paths[path] = as_text(nvme.paths[path]);

    // The view's own header read, so a transport ID names a checkpoint as a file path does.
    auto const [checkpoint, read_status] = nvme_read_metadata<fu::standard_yield_t>(as_text(arguments->path));
    if (failed(read_status)) return propagate_error(read_status, outcome), void();
    usearch_dense_metadata_t geometry {};
    adopt_metadata(checkpoint, &geometry);
    nvme_placement_t const placement {as_text(arguments->path), to_span(paths), nvme.cpu_class, arguments->ram,
                                      arguments->executor};
    usearch_dense_index_t const index = with_variant(geometry, placement, outcome);
    if (!index) return;
    *arguments->index = index;
    propagate_error(usearch_status_success_k, outcome, nullptr);
}

USEARCH_EXPORT void usearch_dense_free(usearch_dense_index_t index) {
    if (!index) return;
    wait_named_calls(index->named_calls);
    delete index;
}

USEARCH_EXPORT void usearch_dense_reserve(usearch_dense_reserve_t* arguments) {
    if (submitted<usearch_dense_reserve>(arguments)) return;
    call_outcome_t outcome(*arguments);
    usearch_dense_index_handle_t const& handle = *arguments->index;
    std::visit(
        [&](auto const* verbs) noexcept {
            propagate_error(verbs->reserve(handle.index, arguments->capacity), outcome);
        },
        handle.tables);
}

USEARCH_EXPORT void usearch_dense_clear(usearch_dense_clear_t* arguments) {
    if (submitted<usearch_dense_clear>(arguments)) return;
    call_outcome_t outcome(*arguments);
    usearch_dense_index_handle_t const& handle = *arguments->index;
    std::visit([&](auto const* verbs) noexcept { propagate_error(verbs->clear(handle.index), outcome); },
               handle.tables);
}

#pragma endregion Lifetime

#pragma region Introspection

USEARCH_EXPORT char const* usearch_dense_metric_name(usearch_dense_metric_t metric) {
    switch (metric) {
    case usearch_dense_metric_inner_k: return "inner";
    case usearch_dense_metric_angular_k: return "angular";
    case usearch_dense_metric_euclidean_k: return "euclidean";
    case usearch_dense_metric_hamming_k: return "hamming";
    case usearch_dense_metric_jaccard_k: return "jaccard";
    default: return "unknown";
    }
}

USEARCH_EXPORT usearch_dense_metric_t usearch_dense_metric_from_name(usearch_bytes_view_t name) {
    std::string_view const spelled {reinterpret_cast<char const*>(name.data), name.length};
    for (int code = usearch_dense_metric_inner_k; code <= usearch_dense_metric_jaccard_k; ++code) {
        usearch_dense_metric_t const metric = static_cast<usearch_dense_metric_t>(code);
        if (spelled == usearch_dense_metric_name(metric)) return metric;
    }
    return usearch_dense_metric_unknown_k;
}

USEARCH_EXPORT void usearch_dense_describe(usearch_dense_describe_t* arguments) {
    if (submitted<usearch_dense_describe>(arguments)) return;
    call_outcome_t outcome(*arguments);
    usearch_dense_index_handle_t const& handle = *arguments->index;
    visit_punned(handle, [&]<vendored_residency_t residency_>(punned_dense_index<residency_> punned) noexcept {
        usearch_dense_index_description_t& description = *arguments->description;
        us::dense::geometry_t const geometry = punned.verbs.geometry(punned.index);
        description.metadata = {usearch_metric_of(punned.verbs.kernel(punned.index)),
                                usearch_scalar_of(punned.verbs.dtype), geometry.dimensions,
                                static_cast<std::size_t>(geometry.connectivity)};
        description.size = punned.verbs.size(punned.index);
        description.capacity = punned.verbs.capacity(punned.index);
        description.memory_usage = punned.verbs.memory_usage(punned.index);
        description.cpu_class = static_cast<usearch_cpu_class_t>(punned.verbs.cpu_class);
        description.gpu_class = static_cast<usearch_gpu_class_t>(punned.verbs.gpu_class);
        description.shards_count = punned.verbs.shards_count(punned.index);
        description.load_path = punned.verbs.load_path(punned.index);
    });
    propagate_error(usearch_status_success_k, outcome, nullptr);
}

USEARCH_EXPORT void usearch_dense_count(usearch_dense_count_t* arguments) {
    if (submitted<usearch_dense_count>(arguments)) return;
    call_outcome_t outcome(*arguments);
    us::utilities::uuid128_t const key = uuid_of(arguments->key);
    *arguments->entries = 0;
    visit_punned(
        *arguments->index, [&]<vendored_residency_t residency_>(punned_dense_index<residency_> punned) noexcept {
            usearch_executor_handle_t call_executor;
            typename punned_dense_index<residency_>::executor_t* const executor = executor_of(
                punned, arguments->executor, call_executor, outcome);
            if (!executor) return;
            // The single key rides the batched lookup, which an NVMe view answers with one scan of
            // its key columns.
            propagate_error(punned.verbs.count(punned.index, {&key, 1}, {arguments->entries, 1}, *executor), outcome);
        });
}

#pragma endregion Introspection

#pragma region Insertion and Search

/** Refuses @p count rows of @p columns values unless that is the dimensions @p index was built
 *  with. */
static bool rejects_columns(usearch_dense_index_handle_t const& index, std::size_t count, std::size_t columns,
                            usearch_outcome_t* outcome) noexcept {
    std::size_t dimensions = 0;
    std::visit([&](auto const* verbs) noexcept { dimensions = verbs->geometry(index.index).dimensions; }, index.tables);
    return count && columns != dimensions &&
           !propagate_error(usearch_status_shape_mismatch_k, outcome, "The rows' width differs from the index's");
}

USEARCH_EXPORT void usearch_dense_insert(usearch_dense_insert_t* arguments) {
    if (submitted<usearch_dense_insert>(arguments)) return;
    call_outcome_t outcome(*arguments);
    usearch_traversal_stats_t* const stats = arguments->stats;
    usearch_keys_view_t const keys = arguments->keys;
    usearch_rows_view_t const vectors = arguments->vectors;
    std::size_t const count = arguments->count;
    if (stats) *stats = {};
    if (!count) return propagate_error(usearch_status_success_k, outcome, nullptr), void();
    if (!keys.data || !vectors.data)
        return propagate_error(usearch_status_invalid_argument_k, outcome, "Keys and vectors must not be null"), void();
    if (keys.stride && keys.stride < sizeof(usearch_key_t))
        return propagate_error(usearch_status_unsupported_layout_k, outcome, "The key stride is shorter than a key"),
               void();
    if (rejects_columns(*arguments->index, count, vectors.columns, outcome)) return;
    visit_punned(*arguments->index, [&]<vendored_residency_t residency_>(
                                        punned_dense_index<residency_> punned) noexcept {
        nk_dtype_t const source_dtype = rows_dtype_of(vectors.scalar, vectors.columns, outcome);
        if (source_dtype == nk_dtype_unknown_k) return;
        usearch_executor_handle_t call_executor;
        typename punned_dense_index<residency_>::executor_t* const executor = executor_of(punned, arguments->executor,
                                                                                          call_executor, outcome);
        if (!executor) return;
        using key_t = typename dense_verbs<residency_>::key_t;
        static_assert(sizeof(key_t) == sizeof(usearch_key_t), "The engine reads the caller's keys in place");
        if (vectors.stride && vectors.stride < us::dense::packed_bytes(source_dtype, vectors.columns))
            return propagate_error(usearch_status_unsupported_layout_k, outcome, "The stride is shorter than one row"),
                   void();
        nk::vector_view<key_t> const uuids {reinterpret_cast<char const*>(keys.data), count, keys.stride};
        progress_ticks_t ticks {arguments->progress, count};
        insert_config_t const config {static_cast<candidates_count_t>(arguments->expansion_factor)};
        expected<traversal_stats_t> const inserted = punned.verbs.insert(punned.index, uuids, rows_of(vectors, count),
                                                                         config, *executor, ticks);
        if (crossed(inserted, stats, outcome)) propagate_error(usearch_status_success_k, outcome, nullptr);
    });
}

USEARCH_EXPORT void usearch_dense_refine(usearch_dense_refine_t* arguments) {
    if (submitted<usearch_dense_refine>(arguments)) return;
    call_outcome_t outcome(*arguments);
    usearch_traversal_stats_t* const stats = arguments->stats;
    if (stats) *stats = {};
    if (arguments->report) *arguments->report = {};
    visit_punned(*arguments->index,
                 [&]<vendored_residency_t residency_>(punned_dense_index<residency_> punned) noexcept {
                     usearch_executor_handle_t call_executor;
                     typename punned_dense_index<residency_>::executor_t* const executor = executor_of(
                         punned, arguments->executor, call_executor, outcome);
                     if (!executor) return;
                     progress_ticks_t ticks {arguments->progress, punned.verbs.size(punned.index)};
                     refine_config_t const config = refine_config_of(arguments->refine);
                     auto const [refined, refined_status] = punned.verbs.refine(punned.index, config, *executor, ticks);
                     accumulate_stats(stats, refined);
                     // A stopped pass still healed, so its report stands beside the cancel.
                     if (failed(refined_status) && refined_status != status_t::cancelled_k)
                         return propagate_error(refined_status, outcome), void();
                     write_refine_report(arguments->report, refined);
                     if (failed(refined_status)) return propagate_error(refined_status, outcome), void();
                     propagate_error(usearch_status_success_k, outcome, nullptr);
                 });
}

USEARCH_EXPORT void usearch_dense_health(usearch_dense_health_t* arguments) {
    if (submitted<usearch_dense_health>(arguments)) return;
    call_outcome_t outcome(*arguments);
    *arguments->reachability = {};
    visit_punned(
        *arguments->index, [&]<vendored_residency_t residency_>(punned_dense_index<residency_> punned) noexcept {
            usearch_executor_handle_t call_executor;
            typename punned_dense_index<residency_>::executor_t* const executor = executor_of(
                punned, arguments->executor, call_executor, outcome);
            if (!executor) return;
            auto const [audit, status] = punned.verbs.health(punned.index, *executor);
            if (failed(status)) return propagate_error(status, outcome), void();
            *arguments->reachability = {audit.audited_nodes, audit.reachable_nodes, audit.sweeps, audit.back_sweeps};
            propagate_error(usearch_status_success_k, outcome, nullptr);
        });
}

USEARCH_EXPORT void usearch_dense_search(usearch_dense_search_t* arguments) {
    if (submitted<usearch_dense_search>(arguments)) return;
    call_outcome_t outcome(*arguments);
    std::size_t const count = arguments->count, wanted = arguments->wanted;
    if (arguments->stats) *arguments->stats = {};
    *arguments->matches = {};
    if (rejects_columns(*arguments->index, count, arguments->queries.columns, outcome)) return;
    predicate_t const predicate = predicate_of(arguments->predicate);
    ranked_hints_view_t hints;
    if (!hints_of_view(arguments->hints, count, hints, outcome)) return;
    visit_punned(*arguments->index, [&]<vendored_residency_t residency_>(
                                        punned_dense_index<residency_> punned) noexcept {
        arena_residency_t const residency = punned.verbs.residency(punned.index, 0);
        if (!accepts_traversal(residency, arguments->traversal, arguments->predicate, outcome)) return;
        usearch_arena_handle_t* const arena = claim_arena(arguments->arena, arguments->outcome, arguments->executor,
                                                          residency, {arguments->queries.data}, outcome);
        if (!arena) return;
        if (!count) return propagate_error(usearch_status_success_k, outcome, nullptr), void();
        if (!wanted || !punned.verbs.size(punned.index)) {
            // No walk serves zero matches or an empty index, whose shards may not have grown on any
            // device yet; the counts are zeros staged on the host.
            auto const [zeros, carve_status] = carve_matches(arena->memory, count, 0);
            if (failed(carve_status)) return propagate_error(carve_status, outcome, "Failed to grow the arena"), void();
            for (std::size_t query = 0; query != count; ++query) zeros.counts[query] = 0;
            auto const [matches, publish_status] = publish_matches(*arena, zeros, count);
            if (failed(publish_status)) return propagate_error(publish_status, outcome), void();
            *arguments->matches = matches;
            return propagate_error(usearch_status_success_k, outcome, nullptr), void();
        }
        // A VRAM arena's matches are carved on its device, where the shards' rankings merge in
        // place.
        auto [outputs, carve_status] = arena->device_memory ? carve_matches(arena->device_memory, count, wanted)
                                                            : carve_matches(arena->memory, count, wanted);
        if (failed(carve_status)) return propagate_error(carve_status, outcome, "Failed to grow the arena"), void();
        bool const searched = match_dense_rows(
            punned, arguments->executor, count, arguments->queries, 1, hints, outputs,
            search_config_of(arguments->expansion_factor, arguments->traversal), arguments->progress, arguments->stats,
            outcome,
            [&](auto rows, ranked_hints_view_t chunk_hints, ranked_matches_span_t matches, search_config_t config,
                auto& executor) noexcept {
                return punned.verbs.search(punned.index, rows, chunk_hints, matches, config, predicate, executor);
            });
        if (!searched) return;
        *arguments->matches = outputs.view();
        propagate_error(usearch_status_success_k, outcome, nullptr);
    });
}

USEARCH_EXPORT void usearch_dense_get(usearch_dense_get_t* arguments) {
    if (submitted<usearch_dense_get>(arguments)) return;
    call_outcome_t outcome(*arguments);
    us::utilities::uuid128_t const key = uuid_of(arguments->key);
    usearch_scalar_t const scalar = arguments->outputs_scalar;
    *arguments->vectors = {};
    if (arguments->written) *arguments->written = 0;
    visit_punned(
        *arguments->index, [&]<vendored_residency_t residency_>(punned_dense_index<residency_> punned) noexcept {
            std::size_t const dimensions = punned.verbs.geometry(punned.index).dimensions;
            nk_dtype_t const target_dtype = rows_dtype_of(scalar, dimensions, outcome);
            if (target_dtype == nk_dtype_unknown_k) return;
            usearch_arena_handle_t* const arena = claim_arena(arguments->arena, arguments->outcome, arguments->executor,
                                                              punned.verbs.residency(punned.index, 0), {}, outcome);
            if (!arena) return;
            usearch_executor_handle_t call_executor;
            typename punned_dense_index<residency_>::executor_t* const executor = executor_of(
                punned, arguments->executor, call_executor, outcome);
            if (!executor) return;
            // The single key rides the batched lookups: they size the outputs, then fill them.
            std::size_t entries = 0;
            if (status_t const counted = punned.verbs.count(punned.index, {&key, 1}, {&entries, 1}, *executor);
                failed(counted))
                return propagate_error(counted, outcome), void();
            if (!entries) return propagate_error(usearch_status_success_k, outcome, nullptr), void();
            std::size_t const row_bytes = us::dense::packed_bytes(target_dtype, dimensions);
            // Every residency casts into the rows itself, a device gathering into its arena.
            std::byte* const rows = arena->device_memory ? carve<std::byte>(arena->device_memory, entries * row_bytes)
                                                         : carve<std::byte>(arena->memory, entries * row_bytes);
            if (!rows)
                return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to grow the arena"), void();
            std::size_t total = 0;
            us::dense::rows_span_t const target {rows, entries, dimensions, row_bytes, target_dtype};
            if (status_t const got = punned.verbs.get(punned.index, {&key, 1}, target, {&total, 1}, entries, *executor);
                failed(got))
                return propagate_error(got, outcome, "Failed to export the rows"), void();
            *arguments->vectors = {rows, row_bytes, dimensions, scalar};
            if (arguments->written) *arguments->written = min_of(total, entries);
            propagate_error(usearch_status_success_k, outcome, nullptr);
        });
}

USEARCH_EXPORT void usearch_dense_search_stream_init(usearch_dense_search_stream_init_t* arguments) {
    call_outcome_t outcome(*arguments);
    *arguments->stream = nullptr;
    propagate_error(usearch_status_unsupported_k, outcome, "No residency streams matches yet");
}

USEARCH_EXPORT void usearch_dense_search_stream_next(usearch_dense_search_stream_next_t* arguments) {
    if (submitted<usearch_dense_search_stream_next>(arguments)) return;
    call_outcome_t outcome(*arguments);
    *arguments->matches = {};
    propagate_error(usearch_status_unsupported_k, outcome, "No residency streams matches yet");
}

USEARCH_EXPORT void usearch_dense_search_stream_free(usearch_dense_search_stream_t) {}

#pragma endregion Insertion and Search

#pragma region Cursors and Absorbing

USEARCH_EXPORT void usearch_dense_cursor_init(usearch_dense_cursor_init_t* arguments) {
    call_outcome_t outcome(*arguments);
    *arguments->cursor = nullptr;
    if (arguments->starts_count && !arguments->starts)
        return propagate_error(usearch_status_invalid_argument_k, outcome, "The starts must name keys"), void();
    usearch_dense_index_handle_t const& handle = *arguments->index;
    usearch_dense_cursor_t const opened = new (std::nothrow) usearch_dense_cursor_handle_t(handle.tables, &handle);
    if (!opened)
        return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to allocate the cursor"), void();
    status_t const placed = std::visit(
        [&](auto const* verbs) noexcept {
            using key_t = us::utilities::uuid128_t;
            static_assert(sizeof(key_t) == sizeof(usearch_key_t), "Start keys cross as the walk reads them");
            std::span<key_t const> const starts {reinterpret_cast<key_t const*>(arguments->starts),
                                                 arguments->starts_count};
            auto const [walk, status] = verbs->cursor_init(handle.index, starts);
            opened->walk = walk;
            return status;
        },
        handle.tables);
    if (failed(placed)) return delete opened, propagate_error(placed, outcome), void();
    *arguments->cursor = opened;
    propagate_error(usearch_status_success_k, outcome, nullptr);
}

USEARCH_EXPORT void usearch_dense_cursor_next(usearch_dense_cursor_next_t* arguments) {
    if (submitted<usearch_dense_cursor_next>(arguments)) return;
    call_outcome_t outcome(*arguments);
    std::size_t const wanted = arguments->wanted;
    *arguments->written = 0;
    *arguments->keys = {};
    if (arguments->links) *arguments->links = {};
    if (!arguments->cursor)
        return propagate_error(usearch_status_invalid_argument_k, outcome, "The cursor must not be null"), void();
    if (arguments->index != arguments->cursor->opened_on)
        return propagate_error(usearch_status_invalid_argument_k, outcome, "The cursor was opened on another index"),
               void();
    fu::standard_atomic_ref<u32_t> const busy {arguments->cursor->busy};
    if (busy.fetch_test_and_set_bit(0, std::memory_order_acquire))
        return propagate_error(usearch_status_contended_k, outcome, "Another pull is still running on the cursor"),
               void();
    // A handle never changes its variant, so it is still the one the cursor opened on.
    visit_punned(
        *arguments->index, [&]<vendored_residency_t residency_>(punned_dense_index<residency_> punned) noexcept {
            using key_t = typename dense_verbs<residency_>::key_t;
            static_assert(sizeof(key_t) == sizeof(usearch_key_t), "Keys land in the arena as the walk reads them");
            usearch_arena_handle_t* const arena = claim_arena(
                arguments->arena, arguments->outcome, arguments->executor,
                staging_residency(arguments->arena, punned.verbs.residency(punned.index, 0)), {}, outcome);
            if (!arena) return;
            if (!wanted) return propagate_error(usearch_status_success_k, outcome, nullptr), void();
            usearch_executor_handle_t call_executor;
            typename punned_dense_index<residency_>::executor_t* const executor = executor_of(
                punned, arguments->executor, call_executor, outcome);
            if (!executor) return;
            usearch_key_t* const keys = carve<usearch_key_t>(arena->memory, wanted);
            if (!keys)
                return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to grow the arena"), void();
            std::span<key_t> const identities {reinterpret_cast<key_t*>(keys), wanted};
            links_outputs_t links {arena->memory};
            if (arguments->links && failed(links.open()))
                return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to grow the arena"), void();
            expected<std::size_t> const read = punned.verbs.cursor_next(
                punned.index, arguments->cursor->walk, identities, arguments->links ? &links : nullptr, *executor);
            if (!read) return propagate_error(read.status, outcome, "Failed to export the keys"), void();
            auto const [published_keys, published_keys_status] = publish(*arena, keys, read.value);
            auto const [published_links, published_links_status] = publish_links(*arena, links);
            if (status_t const drained_status = drained(*arena,
                                                        first_failure({published_keys_status, published_links_status}));
                failed(drained_status))
                return propagate_error(drained_status, outcome, "Failed to upload the keys to the device"), void();
            if (arguments->links) *arguments->links = published_links;
            if (read.value) {
                *arguments->keys = {published_keys, sizeof(usearch_key_t)};
                *arguments->written = read.value;
            }
            propagate_error(usearch_status_success_k, outcome, nullptr);
        });
    busy.clear_bits(1, std::memory_order_release);
}

USEARCH_EXPORT void usearch_dense_cursor_free(usearch_dense_cursor_t cursor) {
    if (!cursor) return;
    wait_named_calls(cursor->named_calls);
    delete cursor;
}

USEARCH_EXPORT void usearch_dense_absorb(usearch_dense_absorb_t* arguments) {
    if (submitted<usearch_dense_absorb>(arguments)) return;
    call_outcome_t outcome(*arguments);
    if (arguments->stats) *arguments->stats = {};
    if (arguments->report) *arguments->report = {};
    if (arguments->count && !arguments->donors)
        return propagate_error(usearch_status_invalid_argument_k, outcome, "The donors must name handles"), void();
    visit_punned(*arguments->index,
                 [&]<vendored_residency_t residency_>(punned_dense_index<residency_> punned) noexcept {
                     usearch_executor_handle_t call_executor;
                     typename punned_dense_index<residency_>::executor_t* const executor = executor_of(
                         punned, arguments->executor, call_executor, outcome);
                     if (!executor) return;
                     growing_buffer<void*> donors {global_allocator};
                     if (!collect_variants(punned, arguments->donors, arguments->count, arguments->index,
                                           "An index cannot absorb itself", donors, outcome))
                         return;
                     std::size_t population = punned.verbs.size(punned.index);
                     for (void* const donor : donors) population += punned.verbs.size(donor);
                     progress_ticks_t ticks {arguments->progress, population};
                     refine_config_t const config = refine_config_of(arguments->refine);
                     auto const [refined, status] = punned.verbs.absorb(punned.index, to_writable_span(donors), config,
                                                                        *executor, ticks);
                     // The engine nulls every donor it consumed; their handles go with them.
                     for (std::size_t position = 0; position != arguments->count; ++position)
                         if (!donors[position]) delete std::exchange(arguments->donors[position], nullptr);
                     accumulate_stats(arguments->stats, refined);
                     // A stopped closing refine still healed: its report stands beside the cancel.
                     if (failed(status) && status != status_t::cancelled_k)
                         return propagate_error(status, outcome), void();
                     write_refine_report(arguments->report, refined);
                     if (failed(status)) return propagate_error(status, outcome), void();
                     propagate_error(usearch_status_success_k, outcome, nullptr);
                 });
}

USEARCH_EXPORT void usearch_dense_compact(usearch_dense_compact_t* arguments) {
    if (submitted<usearch_dense_compact>(arguments)) return;
    call_outcome_t outcome(*arguments);
    if (arguments->stats) *arguments->stats = {};
    if (arguments->report) *arguments->report = {};
    if (arguments->count && !arguments->sources)
        return propagate_error(usearch_status_invalid_argument_k, outcome, "The sources must name handles"), void();
    visit_punned(*arguments->target, [&]<vendored_residency_t residency_>(
                                         punned_dense_index<residency_> punned) noexcept {
        usearch_executor_handle_t call_executor;
        typename punned_dense_index<residency_>::executor_t* const executor = executor_of(punned, arguments->executor,
                                                                                          call_executor, outcome);
        if (!executor) return;
        growing_buffer<void*> sources {global_allocator};
        if (!collect_variants(punned, arguments->sources, arguments->count, arguments->target, nullptr, sources,
                              outcome))
            return;
        std::size_t population = 0;
        for (void* const source : sources) population += punned.verbs.size(source);
        us::dense::compact_config_t config;
        config.shard_capacity = arguments->shard_capacity;
        config.expansion_factor = static_cast<u32_t>(arguments->refine.expansion_factor);
        config.memory_limit = arguments->refine.memory_limit;
        progress_ticks_t ticks {arguments->progress, population};
        auto const [compacted, status] = punned.verbs.compact(punned.index, to_span(sources), config, *executor, ticks);
        // Only a whole compaction consumed every other source; a failure frees none of them.
        if (succeeded(status) || status == status_t::cancelled_k)
            for (std::size_t position = 0; position != arguments->count; ++position)
                if (arguments->sources[position] != arguments->target)
                    delete std::exchange(arguments->sources[position], nullptr);
        accumulate_stats(arguments->stats, compacted.graph);
        if (failed(status) && status != status_t::cancelled_k) return propagate_error(status, outcome), void();
        write_refine_report(arguments->report, compacted.graph);
        if (failed(status)) return propagate_error(status, outcome), void();
        propagate_error(usearch_status_success_k, outcome, nullptr);
    });
}

USEARCH_EXPORT void usearch_dense_refinement_init(usearch_dense_refinement_init_t* arguments) {
    if (submitted<usearch_dense_refinement_init>(arguments)) return;
    call_outcome_t outcome(*arguments);
    *arguments->refinement = nullptr;
    if (!arguments->executor)
        return propagate_error(usearch_status_invalid_argument_k, outcome,
                               "A refinement needs an executor that outlives it"),
               void();
    if (arguments->count && !arguments->donors)
        return propagate_error(usearch_status_invalid_argument_k, outcome, "The donors must name handles"), void();
    if (arguments->count && arguments->receivers_count)
        return propagate_error(usearch_status_invalid_argument_k, outcome,
                               "A refinement absorbs donors or splits into receivers, never both"),
               void();
    if (arguments->receivers_count && (arguments->receivers_count < 2 || !arguments->receivers))
        return propagate_error(usearch_status_invalid_argument_k, outcome, "A split needs two receivers or more"),
               void();
    visit_punned(*arguments->index, [&]<vendored_residency_t residency_>(
                                        punned_dense_index<residency_> punned) noexcept {
        usearch_executor_handle_t call_executor;
        typename punned_dense_index<residency_>::executor_t* const executor = executor_of(punned, arguments->executor,
                                                                                          call_executor, outcome);
        if (!executor) return;
        if (!accepts_dense_variants(punned, arguments->donors, arguments->count, arguments->index,
                                    "An index cannot absorb itself", outcome) ||
            !accepts_dense_variants(punned, arguments->receivers, arguments->receivers_count, arguments->index,
                                    "An index cannot split into itself", outcome))
            return;
        usearch_dense_refinement_t const opened = new (std::nothrow) usearch_dense_refinement_handle_t(
            arguments->index->tables, arguments->index, arguments->executor, arguments->donors, arguments->count);
        if (!opened)
            return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to allocate the refinement"),
                   void();
        split_config_t const config {refine_config_of(arguments->refine), arguments->seed};
        auto const [refinement, status] = punned.verbs.refinement_init(
            punned.index, view_dense_variants(arguments->donors, arguments->count),
            view_dense_variants(arguments->receivers, arguments->receivers_count), config, *executor);
        if (failed(status)) return delete opened, propagate_error(status, outcome), void();
        opened->refinement = refinement;
        *arguments->refinement = opened;
        propagate_error(usearch_status_success_k, outcome, nullptr);
    });
}

USEARCH_EXPORT void usearch_dense_refinement_next(usearch_dense_refinement_next_t* arguments) {
    if (submitted<usearch_dense_refinement_next>(arguments)) return;
    call_outcome_t outcome(*arguments);
    if (arguments->reselected) *arguments->reselected = 0;
    if (arguments->changed) *arguments->changed = 0;
    if (arguments->stats) *arguments->stats = {};
    usearch_dense_refinement_t const refinement = arguments->refinement;
    if (!refinement)
        return propagate_error(usearch_status_invalid_argument_k, outcome, "The refinement must not be null"), void();
    if (arguments->index != refinement->opened_on || arguments->executor != refinement->opened_with)
        return propagate_error(usearch_status_invalid_argument_k, outcome,
                               "The refinement was opened on another index or executor"),
               void();
    visit_punned(
        *arguments->index, [&]<vendored_residency_t residency_>(punned_dense_index<residency_> punned) noexcept {
            usearch_executor_handle_t call_executor;
            typename punned_dense_index<residency_>::executor_t* const executor = executor_of(
                punned, arguments->executor, call_executor, outcome);
            if (!executor) return;
            progress_ticks_t ticks {arguments->progress, punned.verbs.size(punned.index)};
            status_t const ran = punned.verbs.refinement_next(punned.index, refinement->refinement, *executor, ticks);
            // The plan nulls every donor it consumed; their handles go with them.
            for (std::size_t position = 0; position != refinement->count; ++position)
                if (refinement->donors[position] && punned.verbs.refinement_consumed(refinement->refinement, position))
                    delete std::exchange(refinement->donors[position], nullptr);
            // Running totals, so a cancelled or failed call still says how far the pass got.
            refine_report_t const refined = punned.verbs.refinement_running(refinement->refinement);
            if (arguments->reselected) *arguments->reselected = refined.reselected_nodes;
            if (arguments->changed) *arguments->changed = refined.changed_nodes;
            accumulate_stats(arguments->stats, refined);
            if (failed(ran)) return propagate_error(ran, outcome), void();
            propagate_error(usearch_status_success_k, outcome, nullptr);
        });
}

USEARCH_EXPORT void usearch_dense_refinement_finish(usearch_dense_refinement_finish_t* arguments) {
    if (submitted<usearch_dense_refinement_finish>(arguments)) return;
    call_outcome_t outcome(*arguments);
    if (arguments->stats) *arguments->stats = {};
    if (arguments->report) *arguments->report = {};
    // Freed whatever the call answers; a plan dying unfinished reopens its index.
    std::unique_ptr<usearch_dense_refinement_handle_t> const refinement {arguments->refinement};
    if (!refinement)
        return propagate_error(usearch_status_invalid_argument_k, outcome, "The refinement must not be null"), void();
    if (arguments->index != refinement->opened_on || arguments->executor != refinement->opened_with)
        return propagate_error(usearch_status_invalid_argument_k, outcome,
                               "The refinement was opened on another index or executor"),
               void();
    visit_punned(*arguments->index, [&]<vendored_residency_t residency_>(
                                        punned_dense_index<residency_> punned) noexcept {
        usearch_executor_handle_t call_executor;
        typename punned_dense_index<residency_>::executor_t* const executor = executor_of(punned, arguments->executor,
                                                                                          call_executor, outcome);
        if (!executor) return;
        auto const [refined, status] = punned.verbs.refinement_finish(punned.index, refinement->refinement, *executor);
        accumulate_stats(arguments->stats, refined);
        if (failed(status)) return propagate_error(status, outcome), void();
        write_refine_report(arguments->report, refined);
        propagate_error(usearch_status_success_k, outcome, nullptr);
    });
}

USEARCH_EXPORT void usearch_dense_sample(usearch_dense_sample_t* arguments) {
    if (submitted<usearch_dense_sample>(arguments)) return;
    call_outcome_t outcome(*arguments);
    reset_sample(*arguments);
    if (!accepts_sample(*arguments, outcome)) return;
    visit_punned(
        *arguments->index, [&]<vendored_residency_t residency_>(punned_dense_index<residency_> punned) noexcept {
            using key_t = typename dense_verbs<residency_>::key_t;
            usearch_executor_handle_t call_executor;
            typename punned_dense_index<residency_>::executor_t* const executor = executor_of(
                punned, arguments->executor, call_executor, outcome);
            if (!executor) return;
            // The draws land on the arena's host pages wherever it lives; a device index
            // refuses them below.
            usearch_arena_handle_t* const arena = claim_arena(
                arguments->arena, arguments->outcome, arguments->executor, punned.verbs.residency(punned.index, 0),
                {arguments->shown.data, arguments->shown_coordinates.data, arguments->focus.data}, outcome);
            if (!arena) return;
            auto const [inputs, view_status] = view_sample<key_t>(*arguments, arena->memory);
            auto const [staging, carve_status] = carve_sample(arena->memory, arguments->wanted);
            if (failed(view_status) || failed(carve_status))
                return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to grow the arena"), void();
            sample_config_t const config {arguments->seed, arguments->expansion_factor};
            auto const [sampled, status] = punned.verbs.sample(punned.index, inputs, staging.template outputs<key_t>(),
                                                               config, *executor);
            if (failed(status)) return propagate_error(status, outcome), void();
            progress_ticks_t ticks {arguments->progress, arguments->wanted};
            if (sampled.written && ticks(sampled.written, arguments->wanted) == continuation_t::cancel_k)
                return propagate_error(usearch_status_cancelled_k, outcome, "The caller's handler stopped the work"),
                       void();
            publish_sample(*arguments, staging, sampled);
            propagate_error(usearch_status_success_k, outcome, nullptr);
        });
}

USEARCH_EXPORT void usearch_dense_split(usearch_dense_split_t* arguments) {
    if (submitted<usearch_dense_split>(arguments)) return;
    call_outcome_t outcome(*arguments);
    if (arguments->report) *arguments->report = {};
    if (arguments->stats) *arguments->stats = {};
    if (arguments->count < 2 || !arguments->receivers)
        return propagate_error(usearch_status_invalid_argument_k, outcome, "A split needs two receivers or more"),
               void();
    visit_punned(*arguments->index, [&]<vendored_residency_t residency_>(
                                        punned_dense_index<residency_> punned) noexcept {
        usearch_executor_handle_t call_executor;
        typename punned_dense_index<residency_>::executor_t* const executor = executor_of(punned, arguments->executor,
                                                                                          call_executor, outcome);
        if (!executor) return;
        if (!accepts_dense_variants(punned, arguments->receivers, arguments->count, arguments->index,
                                    "An index cannot split into itself", outcome))
            return;
        progress_ticks_t ticks {arguments->progress, punned.verbs.size(punned.index)};
        split_config_t const config {refine_config_of(arguments->refine), arguments->seed};
        auto const [refined, status] = punned.verbs.split(
            punned.index, view_dense_variants(arguments->receivers, arguments->count), config, *executor, ticks);
        accumulate_stats(arguments->stats, refined);
        // A stopped closing refine still healed, so its report stands beside the cancel.
        if (failed(status) && status != status_t::cancelled_k) return propagate_error(status, outcome), void();
        write_refine_report(arguments->report, refined);
        if (failed(status)) return propagate_error(status, outcome), void();
        propagate_error(usearch_status_success_k, outcome, nullptr);
    });
}

#pragma endregion Cursors and Absorbing

#pragma region Linear Algebra

USEARCH_EXPORT void usearch_dense_multiply(usearch_dense_multiply_t* arguments) {
    if (submitted<usearch_dense_multiply>(arguments)) return;
    call_outcome_t outcome(*arguments);
    std::size_t const count = arguments->count;
    *arguments->products = {};
    if (arguments->written) *arguments->written = 0;
    if (arguments->required) *arguments->required = 0;
    if (arguments->stats) *arguments->stats = {};
    if (rejects_columns(*arguments->index, count, arguments->left.columns, outcome)) return;
    predicate_t const predicate = predicate_of(arguments->predicate);
    visit_punned(
        *arguments->index, [&]<vendored_residency_t residency_>(punned_dense_index<residency_> punned) noexcept {
            arena_residency_t const residency = punned.verbs.residency(punned.index, 0);
            if (!accepts_traversal(residency, arguments->traversal, arguments->predicate, outcome)) return;
            if (punned.verbs.kernel(punned.index) != nk_kernel_dot_k)
                return propagate_error(usearch_status_config_mismatch_k, outcome,
                                       "A product is a dot, so the kind must be inner product"),
                       void();
            usearch_arena_handle_t* const arena = claim_arena(arguments->arena, arguments->outcome, arguments->executor,
                                                              residency, {arguments->left.data}, outcome);
            if (!arena) return;
            if (!count) return propagate_error(usearch_status_success_k, outcome, nullptr), void();
            // The scored counts stay on the host, where the written and required totals sum.
            std::size_t* const scored = carve<std::size_t>(arena->memory, count);
            if (!scored)
                return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to grow the arena"), void();
            products_tape_t tape;
            // Every query keeps up to `width` of the pairs its walks scored, from its offset.
            auto walk = [&](std::size_t width, usearch_progress_t progress) noexcept {
                auto const [carved, carve_status] = carve_products_tape(*arena, count, width);
                if (failed(carve_status)) return propagate_error(carve_status, outcome, "Failed to grow the arena");
                tape = carved;
                auto sink = [&](us::dense::rows_view_t chunk_rows,
                                typename punned_dense_index<residency_>::executor_t& executor, search_config_t config,
                                std::size_t resolved, std::size_t taking) noexcept {
                    products_tape_t const window {tape.keys, tape.scores, tape.offsets.subspan(resolved, taking + 1),
                                                  width};
                    return punned.verbs.multiply(punned.index, chunk_rows, window, {scored + resolved, taking}, config,
                                                 predicate, executor);
                };
                return for_each_dense_chunk(punned, arguments->executor, count, arguments->left, 1,
                                            search_config_of(arguments->expansion_factor, arguments->traversal),
                                            progress, arguments->stats, outcome, sink);
            };
            // Keeping every pair needs the widest query first, which only a counting walk can tell.
            std::size_t width = arguments->wanted;
            if (!width) {
                if (!walk(0, usearch_progress_t {})) return;
                for (std::size_t const query_scored : std::span {scored, count}) width = max_of(width, query_scored);
            }
            if (!walk(width, arguments->progress)) return;
            std::size_t written = 0, required = 0;
            for (std::size_t const query_scored : std::span {scored, count})
                written += min_of(query_scored, width), required += query_scored;
            *arguments->products = tape_view_of(tape, written);
            if (arguments->written) *arguments->written = written;
            if (arguments->required) *arguments->required = required;
            propagate_error(usearch_status_success_k, outcome, nullptr);
        });
}

static_assert(sizeof(us::dense::softmax_state_t) == sizeof(usearch_dense_softmax_state_t));

USEARCH_EXPORT void usearch_dense_attend(usearch_dense_attend_t* arguments) {
    if (submitted<usearch_dense_attend>(arguments)) return;
    call_outcome_t outcome(*arguments);
    usearch_dense_softmax_state_t* const carry = arguments->carry;
    std::size_t const count = arguments->count, head_count = max_of<std::size_t>(arguments->head_count, 1);
    usearch_scalar_t const outputs_scalar = arguments->outputs_scalar ? arguments->outputs_scalar
                                                                      : arguments->queries.scalar;
    if (arguments->stats) *arguments->stats = {};
    *arguments->outputs = {};
    if (!arguments->queries.data)
        return propagate_error(usearch_status_invalid_argument_k, outcome, "The queries must name storage"), void();
    std::size_t const columns = arguments->queries.columns;
    nk_dtype_t const outputs_dtype = rows_dtype_of(outputs_scalar, columns, outcome);
    if (outputs_dtype == nk_dtype_unknown_k) return;
    if (count % head_count)
        return propagate_error(usearch_status_shape_mismatch_k, outcome, "The heads must divide the rows"), void();
    if (rejects_columns(*arguments->index, count, columns, outcome)) return;
    std::size_t const row_bytes = us::dense::packed_bytes(outputs_dtype, columns);
    f32_t const requested_scale = arguments->score_scale;
    predicate_t const predicate = predicate_of(arguments->predicate);

    visit_punned(*arguments->index, [&]<vendored_residency_t residency_>(
                                        punned_dense_index<residency_> punned) noexcept {
        arena_residency_t const residency = punned.verbs.residency(punned.index, 0);
        if (!accepts_traversal(residency, arguments->traversal, arguments->predicate, outcome)) return;
        if (punned.verbs.kernel(punned.index) != nk_kernel_dot_k)
            return propagate_error(usearch_status_config_mismatch_k, outcome,
                                   "A softmax weighs logits, so the kind must be inner product"),
                   void();
        if (!arguments->value_lookup)
            return propagate_error(usearch_status_invalid_argument_k, outcome, "The value lookup must be set"), void();
        usearch_arena_handle_t* const arena = claim_arena(arguments->arena, arguments->outcome, arguments->executor,
                                                          residency, {arguments->queries.data, carry}, outcome);
        if (!arena) return;
        if (!count) return propagate_error(usearch_status_success_k, outcome, nullptr), void();
        // A VRAM arena's outputs are carved on its device, where the shards' merge writes them in
        // place.
        std::byte* const output_bytes = arena->device_memory ? carve<std::byte>(arena->device_memory, count * row_bytes)
                                                             : carve<std::byte>(arena->memory, count * row_bytes);
        if (!output_bytes)
            return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to grow the arena"), void();
        value_lookup_t const value_lookup {arguments->value_lookup, arguments->value_lookup_state,
                                           usearch_scalar_of(punned.verbs.dtype), columns};
        us::dense::rows_span_t const outputs {output_bytes, count, columns, row_bytes, outputs_dtype};
        // The engine's carry mirrors the C one, so it writes the caller's states in place.
        std::span<us::dense::softmax_state_t> const carried {reinterpret_cast<us::dense::softmax_state_t*>(carry),
                                                             carry ? count : 0};
        f32_t const score_scale =
            requested_scale > 0 ? requested_scale
                                : 1 / std::sqrt(static_cast<f32_t>(punned.verbs.geometry(punned.index).dimensions));
        auto sink = [&](us::dense::rows_view_t chunk_rows,
                        typename punned_dense_index<residency_>::executor_t& executor, search_config_t config,
                        std::size_t resolved, std::size_t taking) noexcept {
            attention_arguments_t const attention {head_count, value_lookup, score_scale,
                                                   outputs.slice(resolved, taking),
                                                   carried.empty() ? carried : carried.subspan(resolved, taking)};
            return punned.verbs.attend(punned.index, chunk_rows, attention, config, predicate, executor);
        };
        if (!for_each_dense_chunk(punned, arguments->executor, count, arguments->queries, head_count,
                                  search_config_of(arguments->expansion_factor, arguments->traversal),
                                  arguments->progress, arguments->stats, outcome, sink))
            return;
        *arguments->outputs = {output_bytes, row_bytes, columns, outputs_scalar};
        propagate_error(usearch_status_success_k, outcome, nullptr);
    });
}

USEARCH_EXPORT void usearch_dense_maxsim(usearch_dense_maxsim_t* arguments) {
    if (submitted<usearch_dense_maxsim>(arguments)) return;
    call_outcome_t outcome(*arguments);
    std::size_t const count = arguments->count, rows_per_query = arguments->rows_per_query;
    if (arguments->stats) *arguments->stats = {};
    *arguments->matches = {};
    if (arguments->written) *arguments->written = 0;
    if (arguments->required) *arguments->required = 0;
    if (!arguments->query_offsets && !rows_per_query)
        return propagate_error(usearch_status_shape_mismatch_k, outcome, "Every query needs at least one row"), void();
    predicate_t const predicate = predicate_of(arguments->predicate);
    ranked_hints_view_t hints;
    if (!hints_of_view(arguments->hints, count, hints, outcome)) return;
    visit_punned(*arguments->index, [&]<vendored_residency_t residency_>(
                                        punned_dense_index<residency_> punned) noexcept {
        arena_residency_t const residency = punned.verbs.residency(punned.index, 0);
        if (!accepts_traversal(residency, arguments->traversal, arguments->predicate, outcome)) return;
        if (punned.verbs.kernel(punned.index) != nk_kernel_dot_k)
            return propagate_error(usearch_status_config_mismatch_k, outcome,
                                   "Late interaction sums maxima over logits, so the kind must be inner product"),
                   void();
        usearch_arena_handle_t* const arena = claim_arena(arguments->arena, arguments->outcome, arguments->executor,
                                                          residency, {arguments->queries.data}, outcome);
        if (!arena) return;
        if (!count) return propagate_error(usearch_status_success_k, outcome, nullptr), void();
        // Uniform queries become offsets in the arena, so every residency reads one layout.
        std::size_t* const uniform = arguments->query_offsets ? nullptr : carve<std::size_t>(arena->memory, count + 1);
        if (!arguments->query_offsets && !uniform)
            return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to grow the arena"), void();
        for (std::size_t query = 0; uniform && query != count + 1; ++query) uniform[query] = query * rows_per_query;
        std::span<std::size_t const> const query_offsets {arguments->query_offsets ? arguments->query_offsets : uniform,
                                                          count + 1};
        if (!accepts_query_offsets(query_offsets, outcome)) return;
        if (rejects_columns(*arguments->index, query_offsets.back(), arguments->queries.columns, outcome)) return;
        usearch_rows_view_t const& rows = arguments->queries;
        if (!rows.data)
            return propagate_error(usearch_status_invalid_argument_k, outcome, "The rows must name storage"), void();
        nk_dtype_t const source_dtype = rows_dtype_of(rows.scalar, rows.columns, outcome);
        if (source_dtype == nk_dtype_unknown_k) return;
        if (rows.stride && rows.stride < us::dense::packed_bytes(source_dtype, rows.columns))
            return propagate_error(usearch_status_unsupported_layout_k, outcome, "The stride is shorter than one row"),
                   void();
        usearch_executor_handle_t call_executor;
        typename punned_dense_index<residency_>::executor_t* const executor = executor_of(punned, arguments->executor,
                                                                                          call_executor, outcome);
        if (!executor) return;
        // A zero `wanted` keeps every key a query scores, which the tape holds room for: every key
        // under a scan, else every row's breadth on every shard, of the longest query.
        std::size_t const wanted = arguments->wanted;
        std::size_t stride = wanted;
        if (!wanted && arguments->traversal == usearch_traversal_scan_k) stride = punned.verbs.size(punned.index);
        else if (!wanted) {
            std::size_t const connectivity = punned.verbs.geometry(punned.index).connectivity;
            std::size_t const breadth = arguments->expansion_factor ? arguments->expansion_factor
                                                                    : default_expansion_factor(connectivity);
            std::size_t longest = 0;
            for (std::size_t query = 0; query != count; ++query)
                longest = max_of(longest, query_offsets[query + 1] - query_offsets[query]);
            stride = longest * punned.verbs.shards_count(punned.index) * breadth;
        }
        auto const [tape, carve_status] = carve_products_tape(*arena, count, stride);
        // The scored counts and each chunk's rebased offsets stay on the host.
        std::size_t* const scored = carve<std::size_t>(arena->memory, count);
        std::size_t* const rebased = carve<std::size_t>(arena->memory, count + 1);
        if (failed(carve_status) || !scored || !rebased)
            return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to grow the arena"), void();
        search_config_t const config = search_config_of(arguments->expansion_factor, arguments->traversal);
        us::dense::rows_view_t const all = rows_of(rows, query_offsets.back());
        progress_ticks_t ticks {arguments->progress, count};
        std::size_t const step = ticks.slice();
        for (std::size_t first = 0; first != count;) {
            std::size_t const taking = min_of(step, count - first), first_row = query_offsets[first];
            for (std::size_t query = 0; query <= taking; ++query)
                rebased[query] = query_offsets[first + query] - first_row;
            products_tape_t const window {tape.keys, tape.scores, tape.offsets.subspan(first, taking + 1), stride};
            nk::range_t const chunk_range = nk::range(first, first + taking);
            ranked_hints_view_t const chunk_hints = hints.counts.empty()
                                                        ? ranked_hints_view_t {}
                                                        : ranked_hints_view_t {hints.keys(chunk_range, nk::slice),
                                                                               hints.counts.subspan(first, taking)};
            expected<traversal_stats_t> const found = punned.verbs.maxsim(
                punned.index,
                maxsim_queries_view_t {all.slice(first_row, rebased[taking]), {rebased, taking + 1}, chunk_hints},
                window, {scored + first, taking}, wanted, config, predicate, *executor);
            if (!crossed(found, arguments->stats, outcome)) return;
            first += taking;
            if (ticks(first, count) == continuation_t::cancel_k)
                return propagate_error(usearch_status_cancelled_k, outcome, "The handler stopped the call"), void();
        }
        std::size_t written = 0, required = 0;
        for (std::size_t const query_scored : std::span {scored, count})
            written += min_of(query_scored, stride), required += query_scored;
        *arguments->matches = tape_view_of(tape, written);
        if (arguments->written) *arguments->written = written;
        if (arguments->required) *arguments->required = required;
        propagate_error(usearch_status_success_k, outcome, nullptr);
    });
}

#pragma endregion Linear Algebra

#pragma region Serialization

USEARCH_EXPORT void usearch_dense_load_ram(usearch_dense_load_ram_t* arguments) {
    if (submitted<usearch_dense_load_ram>(arguments)) return;
    call_outcome_t outcome(*arguments);
    load_dense_index(as_text(arguments->path), arguments->ram, arguments->connectivity, arguments->executor,
                     arguments->progress, arguments->index, outcome);
}

USEARCH_EXPORT void usearch_dense_load_vram(usearch_dense_load_vram_t* arguments) {
    if (submitted<usearch_dense_load_vram>(arguments)) return;
    call_outcome_t outcome(*arguments);
    load_dense_index(as_text(arguments->path), arguments->vram, 0, arguments->executor, arguments->progress,
                     arguments->index, outcome);
}

USEARCH_EXPORT void usearch_dense_load_ram_buffer(usearch_dense_load_ram_buffer_t* arguments) {
    if (submitted<usearch_dense_load_ram_buffer>(arguments)) return;
    call_outcome_t outcome(*arguments);
    load_dense_index(checkpoint_source_t {as_bytes(arguments->buffer)}, arguments->ram, arguments->connectivity,
                     arguments->executor, arguments->progress, arguments->index, outcome);
}

USEARCH_EXPORT void usearch_dense_load_vram_buffer(usearch_dense_load_vram_buffer_t* arguments) {
    if (submitted<usearch_dense_load_vram_buffer>(arguments)) return;
    call_outcome_t outcome(*arguments);
    // A VRAM `usearch_dense_save_buffer` leaves its checkpoint on the device, where it is read.
    load_dense_index(checkpoint_source_of(as_bytes(arguments->buffer)), arguments->vram, 0, arguments->executor,
                     arguments->progress, arguments->index, outcome);
}

USEARCH_EXPORT void usearch_dense_save(usearch_dense_save_t* arguments) {
    if (submitted<usearch_dense_save>(arguments)) return;
    call_outcome_t outcome(*arguments);
    if (!arguments->path.data || !arguments->path.length)
        return propagate_error(usearch_status_invalid_argument_k, outcome, "The path must not be empty"), void();
    visit_punned(
        *arguments->index, [&]<vendored_residency_t residency_>(punned_dense_index<residency_> punned) noexcept {
            usearch_executor_handle_t call_executor;
            typename punned_dense_index<residency_>::executor_t* const executor = executor_of(
                punned, arguments->executor, call_executor, outcome);
            if (!executor) return;
            progress_ticks_t ticks {arguments->progress, punned.verbs.shards_count(punned.index)};
            propagate_error(punned.verbs.save(punned.index, as_text(arguments->path), *executor, ticks), outcome);
        });
}

USEARCH_EXPORT void usearch_dense_save_buffer(usearch_dense_save_buffer_t* arguments) {
    if (submitted<usearch_dense_save_buffer>(arguments)) return;
    call_outcome_t outcome(*arguments);
    *arguments->monolith = {};
    visit_punned(
        *arguments->index, [&]<vendored_residency_t residency_>(punned_dense_index<residency_> punned) noexcept {
            usearch_arena_handle_t* const arena = claim_arena(arguments->arena, arguments->outcome, arguments->executor,
                                                              punned.verbs.residency(punned.index, 0), {}, outcome);
            if (!arena) return;
            usearch_executor_handle_t call_executor;
            typename punned_dense_index<residency_>::executor_t* const executor = executor_of(
                punned, arguments->executor, call_executor, outcome);
            if (!executor) return;
            progress_ticks_t ticks {arguments->progress, punned.verbs.shards_count(punned.index)};
            save_into_arena(
                *arena, punned.verbs.storage_requirements(punned.index),
                [&](bytes_span_t checkpoint) noexcept {
                    return punned.verbs.save_buffer(punned.index, checkpoint, *executor, ticks);
                },
                *arguments->monolith, outcome);
        });
}

USEARCH_EXPORT void usearch_dense_metadata_read(usearch_dense_metadata_read_t* arguments) {
    if (submitted<usearch_dense_metadata_read>(arguments)) return;
    call_outcome_t outcome(*arguments);
    if (!arguments->path.data || !arguments->path.length)
        return propagate_error(usearch_status_invalid_argument_k, outcome, "The path must not be empty"), void();
    // Through the medium a view reads, since a generic char node maps nothing and reports no size.
    auto const [source, status] = nvme_read_metadata<fu::standard_yield_t>(as_text(arguments->path));
    if (failed(status)) return propagate_error(status, outcome), void();
    adopt_metadata(source, arguments->metadata);
    propagate_error(usearch_status_success_k, outcome, nullptr);
}

USEARCH_EXPORT void usearch_dense_metadata_read_buffer(usearch_dense_metadata_read_buffer_t* arguments) {
    if (submitted<usearch_dense_metadata_read_buffer>(arguments)) return;
    call_outcome_t outcome(*arguments);
    if (!arguments->buffer.data || !arguments->buffer.length)
        return propagate_error(usearch_status_invalid_argument_k, outcome, "The buffer must not be empty"), void();
    auto const [source, status] = safegraphs_read_metadata(as_bytes(arguments->buffer));
    if (failed(status)) return propagate_error(status, outcome), void();
    adopt_metadata(source, arguments->metadata);
    propagate_error(usearch_status_success_k, outcome, nullptr);
}

#pragma endregion Serialization

#pragma region Executors

USEARCH_EXPORT usearch_status_t usearch_outcome_wait(usearch_outcome_t* outcome) {
    return host_outcome_verbs().wait(outcome);
}

USEARCH_EXPORT usearch_status_t usearch_outcome_status(usearch_outcome_t* outcome) {
    return host_outcome_verbs().load(outcome);
}

USEARCH_EXPORT void usearch_arena_init(usearch_arena_init_t* arguments) {
    call_outcome_t outcome(*arguments);
    *arguments->arena = nullptr;
    usearch_arena_t const arena = new (std::nothrow) usearch_arena_handle_t {};
    if (!arena) return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to allocate the arena"), void();
    // A VRAM executor writes its outputs on its vendor's first stream; every other executor
    // computes on the host.
    if (arguments->executor)
        if (gpu_executor_t* const devices = std::get_if<gpu_executor_t>(&arguments->executor->executor)) {
            arena->device_memory = gpu_arena_t::on(*devices);
            arena->residency = vram_residency(static_cast<std::size_t>(arena->device_memory.device()));
            arena->executor = arguments->executor;
        }
    *arguments->arena = arena;
    propagate_error(usearch_status_success_k, outcome, nullptr);
}

USEARCH_EXPORT void usearch_arena_free(usearch_arena_t arena) {
    if (!arena) return;
    wait_named_calls(arena->named_calls);
    delete arena;
}

USEARCH_EXPORT void usearch_arena_describe(usearch_arena_describe_t* arguments) {
    call_outcome_t outcome(*arguments);
    usearch_arena_t const arena = arguments->arena;
    if (!arena || !arguments->description)
        return propagate_error(usearch_status_invalid_argument_k, outcome,
                               "The arena and description must not be NULL"),
               void();
    *arguments->description = {arena->residency.residency, arena_device_ordinal(*arena)};
    propagate_error(usearch_status_success_k, outcome, nullptr);
}

USEARCH_EXPORT void usearch_executor_for_ram(usearch_executor_for_ram_t* arguments) {
    call_outcome_t outcome(*arguments);
    *arguments->executor = nullptr;
    auto const [ceiling, pick_status] = pick_cpu_ceiling(arguments->ram.cpu_class);
    if (failed(pick_status))
        return propagate_error(pick_status, outcome, "The CPU does not admit the pinned class"), void();
    usearch_executor_t const executor = new (std::nothrow) usearch_executor_handle_t {};
    if (!executor)
        return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to allocate the executor"), void();
    if (!spawn_ram_executor(*executor, arguments->threads, ceiling, arguments->ram, outcome))
        return delete executor, void();
    *arguments->executor = executor;
    propagate_error(usearch_status_success_k, outcome, nullptr);
}

USEARCH_EXPORT void usearch_executor_on_pool(usearch_executor_on_pool_t* arguments) {
    call_outcome_t outcome(*arguments);
    *arguments->executor = nullptr;
    propagate_error(usearch_status_unsupported_k, outcome, "Borrowed pools are not wired yet");
}

USEARCH_EXPORT void usearch_executor_for_vram(usearch_executor_for_vram_t* arguments) {
    call_outcome_t outcome(*arguments);
    *arguments->executor = nullptr;
    usearch_vram_options_t const& vram = arguments->vram;
    if (arguments->streams)
        return propagate_error(usearch_status_unsupported_k, outcome, "Borrowed streams are not wired yet"), void();
    // The ABI names USearch ordinals, which the listing resolves to each vendor's own; the spawn
    // keeps one stream per distinct device.
    std::span<gpu_device_t const> const listed = gpu_devices();
    growing_buffer<gpu_device_t> devices {global_allocator};
    if (failed(devices.resize(named_count(vram))))
        return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to list the devices"), void();
    for (std::size_t shard = 0; shard != devices.size(); ++shard) {
        std::size_t const ordinal = named_ordinal(vram, shard);
        if (ordinal >= listed.size())
            return propagate_error(usearch_status_device_k, outcome, "A device ordinal is past the listing"), void();
        devices[shard] = listed[ordinal];
    }
    auto const [gpu_class, shared_status] = shared_gpu_class(vram);
    if (failed(shared_status) && shared_status != status_t::config_mismatch_k)
        return propagate_error(shared_status, outcome), void();
    if (vram.gpu_class != usearch_gpu_class_unknown_k && !gpu_pin_capabilities(gpu_class, vram.gpu_class))
        return propagate_error(usearch_status_unsupported_k, outcome, "The devices do not run the pinned class"),
               void();
    auto [spawned, status] = spawn_gpu_executor(gpu_drivers(), devices, arguments->scratch_limit);
    if (status == status_t::config_mismatch_k)
        return propagate_error(status, outcome, "One executor drives one GPU vendor"), void();
    if (failed(status)) return propagate_error(status, outcome), void();
    usearch_executor_t const executor = new (std::nothrow) usearch_executor_handle_t {};
    if (!executor)
        return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to allocate the executor"), void();
    executor->executor.emplace<gpu_executor_t>(std::move(spawned));
    *arguments->executor = executor;
    propagate_error(usearch_status_success_k, outcome, nullptr);
}

USEARCH_EXPORT void usearch_gpu_devices(usearch_gpu_devices_t* arguments) {
    call_outcome_t outcome(*arguments);
    std::span<gpu_device_t const> const listed = gpu_devices();
    std::size_t const copied = arguments->devices ? min_of(listed.size(), arguments->devices_capacity) : 0;
    for (std::size_t position = 0; position != copied; ++position) {
        gpu_device_t const& device = listed[position];
        usearch_gpu_device_t& entry = arguments->devices[position];
        entry = {static_cast<usearch_gpu_vendor_t>(device.vendor),
                 device.vendor_ordinal,
                 static_cast<usearch_gpu_class_t>(gpu_class_of(device)),
                 device.compute_units,
                 device.memory_bytes,
                 {}};
        std::memcpy(entry.name, device.name, sizeof(entry.name));
    }
    if (arguments->devices_count) *arguments->devices_count = listed.size();
    propagate_error(usearch_status_success_k, outcome, nullptr);
}

USEARCH_EXPORT void usearch_executor_for_nvme(usearch_executor_for_nvme_t* arguments) {
    call_outcome_t outcome(*arguments);
    usearch_nvme_options_t const& nvme = arguments->nvme;
    *arguments->executor = nullptr;
    if (nvme.paths_count && !nvme.paths)
        return propagate_error(usearch_status_invalid_argument_k, outcome, "The paths must name storage"), void();
    auto const [ceiling, pick_status] = pick_cpu_ceiling(nvme.cpu_class);
    if (failed(pick_status))
        return propagate_error(pick_status, outcome, "The CPU does not admit the pinned class"), void();

    // A transport identifier names a namespace the kernel no longer owns, which only a userspace
    // engine reaches, so one executor serves either kernel paths or transport identifiers, never
    // both.
    std::size_t transport_ids = 0, pcie_ids = 0;
    for (std::size_t path = 0; path != nvme.paths_count; ++path) {
        std::string_view const named = as_text(nvme.paths[path]);
        transport_ids += names_transport_id(named);
        pcie_ids += named.starts_with("spdk://") || named.find("trtype:PCIe") != std::string_view::npos;
    }
    if (transport_ids && transport_ids != nvme.paths_count)
        return propagate_error(usearch_status_config_mismatch_k, outcome,
                               "Kernel paths and transport identifiers need different engines"),
               void();
    usearch_nvme_class_t const nvme_class = arguments->nvme_class;
    if (nvme_class > usearch_nvme_class_windows_ioring_k)
        return propagate_error(usearch_status_invalid_argument_k, outcome, "Unknown NVMe class"), void();
    if (nvme_class != usearch_nvme_class_unknown_k &&
        (nvme_class == usearch_nvme_class_linux_spdk_k) != (transport_ids != 0))
        return propagate_error(usearch_status_config_mismatch_k, outcome,
                               "SPDK serves transport identifiers, and the kernel engines serve paths"),
               void();
    // Zero leaves each call's depth to the block queues of the media it reads.
    io_engine_config_t config;
    switch (nvme_class) {
    case usearch_nvme_class_posix_k: config.transport = io_transport_t::blocking_k; break;
    case usearch_nvme_class_linux_aio_k: config.transport = io_transport_t::linux_aio_k; break;
    case usearch_nvme_class_linux_uring_k: config.transport = io_transport_t::linux_uring_k; break;
    case usearch_nvme_class_linux_uring_passthrough_k: config.transport = io_transport_t::linux_uring_nvme_k; break;
    case usearch_nvme_class_windows_ioring_k: config.transport = io_transport_t::windows_ioring_k; break;
    default: break;
    }
    if (transport_ids) {
#if USEARCH_WITH_LINUX_SPDK
        // Fabrics copy every byte through a socket, so only PCIe controllers need hugepages to DMA
        // into.
        linux_spdk_process_t::config_t environment;
        environment.memory = pcie_ids ? linux_spdk_memory_t::hugepages_k : linux_spdk_memory_t::plain_pages_k;
        if (status_t const acquired = linux_spdk_process_t::acquire<fu::standard_yield_t>(environment);
            failed(acquired))
            return propagate_error(acquired, outcome, "The SPDK environment did not start"), void();
        config.transport = io_transport_t::linux_spdk_k;
#else
        (void)pcie_ids;
        return propagate_error(usearch_status_unsupported_k, outcome, "Transport identifiers need an SPDK build"),
               void();
#endif
    }
    usearch_executor_t const executor = new (std::nothrow) usearch_executor_handle_t {};
    if (!executor)
        return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to allocate the executor"), void();
    if (!spawn_nvme_executor(*executor, arguments->threads, {nvme.paths, nvme.paths_count}, config, ceiling, outcome))
        return delete executor, void();
    *arguments->executor = executor;
    propagate_error(usearch_status_success_k, outcome, nullptr);
}

USEARCH_EXPORT void usearch_executor_drain(usearch_executor_drain_t* arguments) {
    call_outcome_t outcome(*arguments);
    // A null executor runs every call to completion before returning, so nothing waits behind it.
    if (!arguments->executor) return propagate_error(usearch_status_success_k, outcome, nullptr), void();
    usearch_executor_handle_t& executor = *arguments->executor;
    if (runs_holding_call(executor))
        return propagate_error(usearch_status_contended_k, outcome, "A callback cannot drain its executor"), void();
    await_delegated_call(executor);
    wait_while_claimed(executor);
    // Threads spin after every dispatch already, so only parking needs a word to them.
    if (forkunion_executor_t* const threads = threads_of(executor);
        threads && arguments->mood != usearch_mood_spinning_k)
        threads->park();
    propagate_error(visit_executor(executor, [](auto& typed) noexcept { return typed.wait(); }), outcome);
}

USEARCH_EXPORT void usearch_executor_free(usearch_executor_t executor) {
    if (!executor) return;
    await_delegated_call(*executor);
    wait_while_claimed(*executor);
    if (nvme_executor_t* const nvme = std::get_if<nvme_executor_t>(&executor->executor)) nvme->release();
    delete executor;
}

#pragma endregion Executors

} // extern "C"
