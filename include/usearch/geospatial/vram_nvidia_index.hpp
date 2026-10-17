/**
 *  @file include/usearch/geospatial/vram_nvidia_index.hpp
 *  @author Ash Vardanian
 *  @date October 15, 2026
 *  @brief Similarity Search over observations on NVIDIA GPUs: the basic VRAM index under the kit
 *      @c geospatial_ladder picks in @c vram_nvidia_metric.cuh, and the record of the kernel
 *      judging each walk's matches.
 *
 *  Plain C++, so the host classes build with any compiler: the members reading the kit, which pick
 *  a launch's rung and size its spans, are defined beside the kit in the metric, which only the
 *  modality's NVIDIA unit compiles, as is the judging kernel. Rows are the 32-byte
 *  @c earth_observation_t, in the caller's tape and in the index alike.
 */
#pragma once

#include <cstddef> // `std::size_t`

#include <numkong/capabilities.h> // `nk_kernel_sqeuclideans3d_k` - the metric the kit scores under

#include "usearch/utilities/types.hpp"
#include "usearch/basic/vram_nvidia_index.hpp"
#include "usearch/geospatial/ram_index.hpp"

namespace unum::usearch::geospatial {
using namespace unum::usearch::utilities;

/** Similarity Search over observations on NVIDIA GPUs, under the squared chord the kit
 *  @c geospatial_ladder picks under @p capability_ scores: a lane scores each candidate, a warp
 *  walks each query, and rows keep their neighbors by tangent-plane cone. Node IDs are
 *  @p node_id_type_, and @p kernels_type_ names the fatbin. */
template <typename node_id_type_, nk_capability_t capability_, typename kernels_type_>
struct vram_nvidia_index
    : basic::vram_index<vram_nvidia_index<node_id_type_, capability_, kernels_type_>, node_id_type_> {

    using base_t = basic::vram_index<vram_nvidia_index, node_id_type_>;

    /** Rungs of the kit's kernels: one, serving every observation. */
    inline static constexpr std::size_t rungs_k = 1;

    /** Pins the index to the storage allocator's device. */
    explicit vram_nvidia_index(cuda_virtual_memory_allocator<byte_t> const& allocator = {}) noexcept
        : base_t(nk_kernel_sqeuclideans3d_k, allocator) {}

  private:
    friend base_t;

    /** The rung serving rows of @p longest bytes, or @c rungs_k when none does. Defined in
     *  @c vram_nvidia_metric.cuh, as are the members below reading the kit. */
    static std::size_t rung_serving_(std::size_t longest) noexcept;

    /** Lanes the kit scores a row with on @p rung. */
    static u32_t lanes_per_row_(std::size_t rung) noexcept;

    /** Bytes the row at @p data takes once prepared. */
    static std::size_t span_bytes_(byte_t const* data, std::size_t bytes) noexcept;

    /** Bytes the @p stored observation saves into: its own 32, as the RAM codec saves it. */
    static expected<std::size_t> saved_bytes_(bytes_view_t stored) noexcept {
        return {stored.size(), status_t::success_k};
    }

    /** Saves the @p stored observation into @p saved as it is. */
    static status_t save_value_(bytes_view_t stored, bytes_span_t saved) noexcept { return copy_bytes(saved, stored); }

    /** Bytes the @p saved observation stores into: its own, @c malformed_k unless one whole. */
    static expected<std::size_t> stored_bytes_(bytes_view_t saved) noexcept {
        if (saved.size() != sizeof(earth_observation_t)) return {0, status_t::malformed_k};
        return {saved.size(), status_t::success_k};
    }

    /** Stores the @p saved observation into @p stored as it is. */
    static status_t load_value_(bytes_view_t saved, bytes_span_t stored) noexcept { return copy_bytes(stored, saved); }

    /** Loads the fatbin onto @p device and finds every kernel by its symbol. */
    status_t load_kernels_(int device) noexcept;

    cuda_function_handle_t kernel_function_(std::size_t rung, vram_kernel_t kernel, u32_t slots) const noexcept {
        return kernels_[rung][static_cast<u32_t>(kernel)][gpu_rung_index(slots)];
    }

    /** The fatbin on this device, and its kernels in @c vram_kernel_t order. */
    cuda_module_t module_;
    cuda_function_handle_t kernels_[rungs_k][static_cast<u32_t>(vram_kernel_t::count_k)][gpu_rungs_per_kernel_k] = {};
};

/** One launch of @c cuda_admit, which keeps the first @c wanted entries of each walk's pool that
 *  the relation admits, ascending, beside their chords in metres and their count. */
template <typename node_id_type_>
struct cuda_admit_arguments {
    basic::vram_ragged_storage<node_id_type_> storage;

    /** The queries, @b [queries], and the pools their walks landed, @b [queries,pool] IDs. */
    earth_observation_t const* queries = nullptr;
    node_id_type_ const* pool_ids = nullptr;
    u32_t queries_count = 0;
    u32_t pool = 0;
    u32_t wanted = 0;

    relation_t relation = relation_t::near_k;
    f64_t radius_metres = 0;
    f64_t slack_milliseconds = 0;

    /** The landing, @b [queries,wanted] IDs and metres, and a count per query. */
    node_id_type_* matched_ids = nullptr;
    f64_t* matched_metres = nullptr;
    u64_t* matched_counts = nullptr;
};

} // namespace unum::usearch::geospatial
