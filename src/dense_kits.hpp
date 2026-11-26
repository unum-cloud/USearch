/**
 *  @file src/dense_kits.hpp
 *  @author Ash Vardanian
 *  @date August 14, 2026
 *  @brief What the dense GPU variants run on: the storage dtypes this build compiles, the value
 *      lookup a walk calls, and each GPU class's device code per dtype.
 *
 *  The class units, the device kernel units, the tests and the benchmark name the same types
 *  through it, so none of them restates a dtype list or a fatbin.
 */
#pragma once

#include <cstddef> // `std::size_t`

#include <array> // `std::array` - a scored batch's keys and rows, a list's dtypes
#include <bit>   // `std::bit_cast` - the C key struct & `uuid128_t` share sixteen bytes
#include <span>  // `std::span` - a scored batch, the bounds of a class's device code

#include <numkong/types.h>    // `nk_dtype_t`
#include <numkong/vector.hpp> // `nk::vector_view` - the rows a value lookup resolves

#include "usearch/dense_index.h"       // `usearch_value_lookup_t`
#include "usearch/utilities/types.hpp" // `uuid128_t`, `bytes_view_t`
#include "usearch/dense/types.hpp"     // `dense::quads_ladder` & kin - the CPU kit picks

#include "classes.hpp" // `gpu_class_t`, `numkong_capabilities_of`

#pragma region Type Lists

/** How many storage dtypes the dense variants cover, from the front of the ladder bf16, u1, i8,
 *  e4m3, f16, u8, e5m2, i4, u4, e3m2, e2m3. CMake passes it, so only the builds bypassing CMake
 *  take this default. */
#if !defined(USEARCH_DENSE_DTYPES)
#define USEARCH_DENSE_DTYPES 4
#endif
#if USEARCH_DENSE_DTYPES < 1 || USEARCH_DENSE_DTYPES > 11
#error "USEARCH_DENSE_DTYPES must be an integer from 1 to 11"
#endif

/** Storage dtypes in ladder order, the one list every dense class unit folds its variants over. */
template <nk_dtype_t... dtypes_>
struct dtype_list {
    static constexpr std::size_t size_k = sizeof...(dtypes_);
    static constexpr std::array<nk_dtype_t, sizeof...(dtypes_)> dtypes_k {dtypes_...};

    /** The index of @p dtype in the list, or @c size_k when it is absent. */
    static constexpr std::size_t index_of(nk_dtype_t dtype) noexcept {
        std::size_t position = 0;
        while (position != size_k && dtypes_k[position] != dtype) ++position;
        return position;
    }
};

/** Takes dtypes off the front of a pack into @p taken_type_ until @p count_ are taken. */
template <std::size_t count_, typename taken_type_, nk_dtype_t... rest_>
struct first_dtypes_of {
    using type = taken_type_;
};
template <std::size_t count_, nk_dtype_t... taken_, nk_dtype_t next_, nk_dtype_t... rest_>
    requires(count_ != 0)
struct first_dtypes_of<count_, dtype_list<taken_...>, next_, rest_...> {
    using type = typename first_dtypes_of<count_ - 1, dtype_list<taken_..., next_>, rest_...>::type;
};

/** The leading @p count_ dtypes of @p list_type_, as a @c dtype_list. */
template <std::size_t count_, typename list_type_>
struct first_dtypes;
template <std::size_t count_, nk_dtype_t... dtypes_>
struct first_dtypes<count_, dtype_list<dtypes_...>> {
    using type = typename first_dtypes_of<count_, dtype_list<>, dtypes_...>::type;
};

/** Every dense storage dtype, in the order the variants cover them from the front. */
using dense_ladder_dtypes_t = dtype_list<nk_bf16_k, nk_u1_k, nk_i8_k, nk_e4m3_k, nk_f16_k, nk_u8_k, nk_e5m2_k, nk_i4_k,
                                         nk_u4_k, nk_e3m2_k, nk_e2m3_k>;

/** The dense storage dtypes this build compiles variants for. */
using dense_dtypes_t = typename first_dtypes<USEARCH_DENSE_DTYPES, dense_ladder_dtypes_t>::type;

#pragma endregion Type Lists

#pragma region Kits

/** The C value lookup lifted onto the index's key vocabulary: one callback per scored batch, at
 *  the batch's static extent, a NULL row answering an empty view that refuses the call. A GPU walk
 *  takes it as bytes and calls it one key at a time on the device. */
struct value_lookup_t {
    usearch_value_lookup_t callback;
    void* callback_state;
    usearch_scalar_t scalar;
    std::size_t dimensions;

    template <typename scalar_type_, std::size_t extent_>
    void operator()(std::span<util::uuid128_t const, extent_> keys,
                    std::span<nk::vector_view<scalar_type_>, extent_> rows) const noexcept {
        std::array<usearch_key_t, extent_> batch;
        std::array<void const*, extent_> found;
        for (std::size_t lane = 0; lane != extent_; ++lane) batch[lane] = std::bit_cast<usearch_key_t>(keys[lane]);
        callback(callback_state, batch.data(), extent_, scalar, dimensions, found.data());
        for (std::size_t lane = 0; lane != extent_; ++lane)
            rows[lane] = found[lane]
                             ? nk::vector_view<scalar_type_> {static_cast<scalar_type_ const*>(found[lane]), dimensions}
                             : nk::vector_view<scalar_type_> {};
    }
};

/** The start and end of every dtype's device code one GPU class unit embeds, in pairs in
 *  @c dense_dtypes_t order, as the table CMake writes for the class, which each VRAM class unit
 *  specializes for its own class. */
template <gpu_class_t gpu_class_>
std::span<util::byte_t const* const> bundle_bounds() noexcept;

/** The device code of one GPU class and dtype, between the bounds of the dtype's position. */
template <gpu_class_t gpu_class_, nk_dtype_t dtype_>
inline static util::bytes_view_t bundled_code() noexcept {
    std::span<util::byte_t const* const> const bounds = bundle_bounds<gpu_class_>();
    std::size_t const position = dense_dtypes_t::index_of(dtype_);
    return util::bytes_view_t {bounds[2 * position], bounds[2 * position + 1]};
}

/** One NVIDIA class's fatbin of one dtype's kernels. */
template <gpu_class_t gpu_class_, nk_dtype_t dtype_>
struct dense_nvidia_fatbin {
    inline static constexpr nk_capability_t capability_k = numkong_capabilities_of(gpu_class_);
    static util::bytes_view_t bytes() noexcept { return bundled_code<gpu_class_, dtype_>(); }
};

/** One AMD class's code object of one dtype's kernels, an offload bundle of its listed
 *  processors. */
template <gpu_class_t gpu_class_, nk_dtype_t dtype_>
struct dense_amd_code_object {
    inline static constexpr nk_capability_t capability_k = numkong_capabilities_of(gpu_class_);
    static util::bytes_view_t bytes() noexcept { return bundled_code<gpu_class_, dtype_>(); }
};

/** The kits one CPU class runs a dtype on, each the pick of its own ladder under the class's
 *  NumKong mask; @c ram_metric.hpp supplies the rungs. */
template <nk_dtype_t dtype_, cpu_class_t cpu_class_>
using quads_kit_of =
    typename us::dense::quads_ladder<typename nk::type_for<dtype_>::type, numkong_capabilities_of(cpu_class_)>::kit_t;
template <nk_dtype_t dtype_, cpu_class_t cpu_class_>
using tiles_kit_of =
    typename us::dense::tiles_ladder<typename nk::type_for<dtype_>::type, numkong_capabilities_of(cpu_class_)>::kit_t;
template <cpu_class_t cpu_class_>
using elementwise_kit_of = typename us::dense::elementwise_ladder<numkong_capabilities_of(cpu_class_)>::kit_t;
template <nk_dtype_t dtype_, cpu_class_t cpu_class_>
using rmscast_kit_of =
    typename us::dense::rmscast_ladder<typename nk::type_for<dtype_>::type, numkong_capabilities_of(cpu_class_)>::kit_t;
template <cpu_class_t cpu_class_>
using visits_kit_of = typename us::dense::visits_hash_set_ladder<numkong_capabilities_of(cpu_class_)>::kit_t;

#pragma endregion Kits
