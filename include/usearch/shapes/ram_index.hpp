/**
 *  @file include/usearch/shapes/ram_index.hpp
 *  @author Ash Vardanian
 *  @date October 13, 2026
 *  @brief Similarity Search over 3D point clouds of any size, like molecules, scans and meshes.
 *
 *  A stored value is a NumKong pack of one cloud: centered, rotated onto its principal axes, and
 *  kept with the constants that bound its distances. Each node record owns its pack, built from the
 *  index's allocator, so a value costs one block and no tape. Scoring is batched: one @c reset →
 *  @c push → @c drain round becomes a single 1×M kernel call over the whole graph fan-out, through
 *  the metric of @c ram_metric.hpp.
 *
 *  @section shapes_ram_index_persistence Persistence
 *
 *  Checkpoints are SafeGraphs monoliths, written by @c safegraphs_write_index and read by
 *  @c safegraphs_read_index with @ref codec, under the kernel's token: @c chamfer, @c hausdorff,
 *  @c wasserstein_procrustes or @c gromov_wasserstein. The value column is a ragged @c f16 or
 *  @c f32 matrix of one @b [points,3] slice per node, holding each cloud in its inserted pose
 *  within one storage rounding. A load packs every cloud again, so distances after a reload agree
 *  with those before it to the storage precision, rather than bit for bit.
 */
#pragma once

#include <cstddef> // `std::size_t`

#include <array>       // `std::array` - the one cloud a load packs
#include <memory>      // `std::allocator_traits` - the values' rebound allocator
#include <span>        // `std::span` - the coordinates a save writes
#include <type_traits> // `std::is_same_v` - the codec's dtype
#include <utility>     // `std::move`

#include <forkunion/atomics.hpp> // `fu::standard_atomic_ref` - the default reference

#include <numkong/shapes.hpp> // `nk::shapes_packed`
#include <numkong/tensor.hpp> // `nk::matrix_view` - a row's borrowed cloud

#include "usearch/utilities/types.hpp"
#include "usearch/utilities/ram_device.hpp"        // `forkunion_allocator` - the default placement
#include "usearch/utilities/safegraphs_format.hpp" // `matrix_shape_t` - the value column's shape
#include "usearch/basic/ram_index.hpp"
#include "usearch/shapes/ram_metric.hpp"

namespace unum::usearch::shapes {

/** One stored cloud: a pack of exactly one, owning its block through @p allocator_type_. */
template <typename scalar_type_, typename allocator_type_ = forkunion_allocator<char>>
using value = nk::shapes_packed<scalar_type_, allocator_type_>;

static_assert(default_alignment_k >= nk_default_alignment_k, "Values would start packs off NumKong's boundary");

/** Clouds out in their inserted pose; clouds in packed again, through the record's own allocator.
 *  Every backend packs as the serial kit does, so a pack moves between classes and towers as is,
 *  and one pose conversion serves the RAM and GPU indexes alike. */
template <typename kit_type_>
struct codec {
    using scalar_t = typename kit_type_::scalar_t;
    using raw_t = typename scalar_t::raw_t;
    using pack_kit_t = typename shapes_ladder<scalar_t, nk_cap_serial_k>::kit_t;
    inline static constexpr dtype_t value_dtype_k = std::is_same_v<scalar_t, nk::f16_t> ? dtype_t::f16_k
                                                                                        : dtype_t::f32_k;
    inline static constexpr column_shape_t value_shape_k = matrix_shape_t {0, 3};

    template <typename value_type_>
    std::size_t storage_requirements(value_type_ const& value) const noexcept {
        return value.max_points() * 3 * sizeof(scalar_t);
    }

    /** The points of the one cloud @p pack holds into @p pose in their inserted pose; @p pose holds
     *  exactly those points. */
    static status_t save_pack(bytes_view_t pack, bytes_span_t pose) noexcept {
        nk_size_t points = 0;
        if (status_t const read = status_of(
                pack_kit_t::packed_shape(pack.data(), &points, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr));
            failed(read))
            return read;
        if (pose.size() != points * 3 * sizeof(raw_t)) return status_t::config_mismatch_k;
        if (pose.empty()) return status_t::success_k;
        return status_of(pack_kit_t::unpack(pack.data(), reinterpret_cast<raw_t*>(pose.data()), nullptr));
    }

    /** The cloud @p pose holds packed into @p pack, which the kit's @c pack_size sized. */
    static status_t load_pack(bytes_view_t pose, bytes_span_t pack) noexcept {
        constexpr std::size_t point_bytes_k = 3 * sizeof(raw_t);
        if (pose.size() % point_bytes_k) return status_t::malformed_k;
        return status_of(pack_kit_t::pack(reinterpret_cast<raw_t const*>(pose.data()), pose.size() / point_bytes_k,
                                          pack.data(), nullptr));
    }

    template <typename value_type_>
    status_t save(value_type_ const& value, bytes_span_t output) const noexcept {
        std::size_t const pose_bytes = storage_requirements(value);
        if (output.size() < pose_bytes) return status_t::overflow_k;
        if (value.empty()) return status_t::success_k;
        nk_size_t pack_bytes = 0;
        if (status_t const sized = status_of(pack_kit_t::pack_size(value.max_points(), &pack_bytes)); failed(sized))
            return sized;
        return save_pack({static_cast<byte_t const*>(value[0]), pack_bytes}, output.first(pose_bytes));
    }

    template <typename value_type_>
    status_t load(bytes_view_t input, value_type_& value) const noexcept {
        std::array const clouds {elements_of<scalar_t>(input)};
        auto [packed, status] = value_type_::make(clouds, value.get_allocator(), pack_kit_t::pack_size,
                                                  pack_kit_t::pack);
        if (nk::failed(status)) return status_of(status);
        value = std::move(packed);
        return status_t::success_k;
    }

    /** What @ref view hands out for a stored row. */
    using view_t = nk::matrix_view<scalar_t>;

    /** The @b [points,3] cloud a stored @p row holds, borrowed: every whole point its bytes
     *  hold. */
    view_t view(bytes_view_t row) const noexcept {
        std::span<scalar_t const> const coordinates = elements_of<scalar_t>(row);
        return nk::matrix_view<scalar_t>(coordinates.data(), coordinates.size() / 3, 3);
    }
};

/**
 *  @brief The shapes RAM index: the basic graph over packs of @p kit_type_ 's scalar, scored in its
 *      distance type.
 *  @tparam allocator_type_ Places the node records, the neighbor rows and the key map, and,
 *      rebound, every value.
 */
template <typename kit_type_, typename key_type_ = uuid_t, typename wait_type_ = dummy_waiter_t,
          template <typename> class atomic_ref_template_ = fu::standard_atomic_ref,
          typename allocator_type_ = forkunion_allocator_t, typename cache_hints_type_ = dummy_cache_hints_t>
using ram_index = basic::ram_index<
    value<typename kit_type_::scalar_t, typename std::allocator_traits<allocator_type_>::template rebind_alloc<char>>,
    key_type_, u32_t, typename kit_type_::distance_t, wait_type_, atomic_ref_template_, allocator_type_,
    cache_hints_type_>;

} // namespace unum::usearch::shapes
