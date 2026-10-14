/**
 *  @file include/usearch/utilities/vram_device.hpp
 *  @author Ash Vardanian
 *  @date July 22, 2026
 *  @brief The vendor-neutral face of the GPU residency: the process-wide device listing every
 *      VRAM call numbers its devices by, one executor and one arena over every vendor.
 *
 *  One binary drives NVIDIA, AMD and Apple GPUs at once, each vendor through a runtime loaded on
 *  first use and declared without its SDK, so every unit names all three: NVIDIA devices come
 *  first, then AMD, then Apple, and a device's position in the listing is its USearch ordinal. A
 *  vendor whose runtime is absent contributes no devices; one whose kernels this build lacks still
 *  lists them, and placing an index there refuses.
 *
 *  Every vendor's device header provides the same surface: @c <vendor>_gpu_devices, an executor
 *  with a static @c spawn over vendor ordinals, @c device, @c vendor_k and @c device_arena_t, and
 *  that arena, made from the executor's first stream, with @c allocate, @c contains, @c clear,
 *  @c upload and @c drain.
 */
#pragma once

#include <cstddef> // `std::size_t`

#include <mutex>   // `std::call_once` - the listing is taken once per process
#include <span>    // `std::span` - the listing handed out
#include <variant> // `std::variant`, `std::monostate` - one alternative per vendor

#include "usearch/utilities/ram_buffers.hpp"        // `growing_buffer` - the ordinals an executor opens over
#include "usearch/utilities/types.hpp"              // `gpu_device_t`
#include "usearch/utilities/vram_amd_device.hpp"    // `amd_gpu_devices`, `hip_executor_t`
#include "usearch/utilities/vram_apple_device.hpp"  // `apple_gpu_devices`, `metal_executor_t`
#include "usearch/utilities/vram_nvidia_device.hpp" // `nvidia_gpu_devices`, `cuda_executor_t`

namespace unum::usearch::utilities {

/** A variant over @c std::monostate, the state of no GPU, then each vendor's @p selector_type_
 *  applied to its executor, in listing order. */
template <template <typename> typename selector_type_>
using gpu_vendors_variant =
    std::variant<std::monostate, typename selector_type_<cuda_executor_t>::type,
                 typename selector_type_<hip_executor_t>::type, typename selector_type_<metal_executor_t>::type>;

/** Visits each vendor's executor type in listing order until @p visitor answers true. */
template <typename visitor_type_>
bool for_each_gpu_vendor(visitor_type_&& visitor) noexcept {
    return visitor(std::type_identity<cuda_executor_t> {}) || visitor(std::type_identity<hip_executor_t> {}) ||
           visitor(std::type_identity<metal_executor_t> {});
}

template <typename executor_type_>
struct gpu_executor_of {
    using type = executor_type_;
};

template <typename executor_type_>
struct gpu_device_arena_of {
    using type = typename executor_type_::device_arena_t;
};

/** Whichever vendor's executor a VRAM call runs on, or none. */
using gpu_executor_t = gpu_vendors_variant<gpu_executor_of>;

/** Every GPU this process can drive: NVIDIA devices first, then AMD, then Apple, listed once. A
 *  vendor whose runtime fails to load contributes none. */
inline std::span<gpu_device_t const> gpu_devices() noexcept {
    static gpu_device_t listing[3 * gpu_devices_capacity_k];
    static std::size_t count = 0;
    static std::once_flag listed;
    std::call_once(listed, [] {
        std::span<gpu_device_t const> const vendors[] = {nvidia_gpu_devices(), amd_gpu_devices(), apple_gpu_devices()};
        for (std::span<gpu_device_t const> vendor : vendors)
            for (gpu_device_t const& device : vendor) listing[count++] = device;
    });
    return {listing, count};
}

/** Spawns the executor of the one vendor every device in @p devices belongs to, one stream per
 *  distinct device over their vendor ordinals, which every shard there shares:
 *  @c config_mismatch_k when they span vendors, @c device_error_k when none is named or their
 *  runtime refuses. */
inline expected<gpu_executor_t> spawn_gpu_executor(std::span<gpu_device_t const> devices) noexcept {
    if (devices.empty()) return {{}, status_t::device_error_k};
    growing_buffer<int> ordinals {global_allocator};
    if (failed(ordinals.reserve_exactly(devices.size()))) return {{}, status_t::bad_alloc_k};
    for (gpu_device_t const& device : devices) {
        if (device.vendor != devices[0].vendor) return {{}, status_t::config_mismatch_k};
        int const ordinal = static_cast<int>(device.vendor_ordinal);
        bool opened = false;
        for (int const earlier : ordinals) opened |= earlier == ordinal;
        if (!opened) ordinals.push_back_reserved(ordinal);
    }
    expected<gpu_executor_t> result {{}, status_t::device_error_k};
    for_each_gpu_vendor([&]<typename executor_type_>(std::type_identity<executor_type_>) noexcept {
        if (executor_type_::vendor_k != devices[0].vendor) return false;
        auto [spawned, status] = executor_type_::spawn(to_span(ordinals));
        if (succeeded(status)) result.value = std::move(spawned);
        result.status = status;
        return true;
    });
    return result;
}

/** Device memory a C-layer call lands its outputs in, on the vendor and first stream of the
 *  executor it was made for, or none for host-resident calls; it serves calls on that executor
 *  alone, and dies before it. */
struct gpu_arena_t {
    gpu_vendors_variant<gpu_device_arena_of> arena;

    /** Places the arena on @p executor 's vendor and first stream; none for no GPU. */
    static gpu_arena_t on(gpu_executor_t& executor) noexcept {
        gpu_arena_t placed;
        std::visit(
            [&]<typename executor_type_>(executor_type_& typed) noexcept {
                if constexpr (!std::is_same_v<executor_type_, std::monostate>)
                    placed.arena.template emplace<typename executor_type_::device_arena_t>(typed.stream(0));
            },
            executor);
        return placed;
    }

    /** Whether the arena lives on a device rather than nowhere. */
    explicit operator bool() const noexcept { return !std::holds_alternative<std::monostate>(arena); }

    /** The vendor ordinal of the arena's device, or -1 for none. */
    int device() const noexcept {
        return std::visit(
            []<typename arena_type_>(arena_type_ const& typed) noexcept {
                if constexpr (std::is_same_v<arena_type_, std::monostate>) return -1;
                else return typed.device();
            },
            arena);
    }

    /** Carves @p count values on the device, or null when it cannot grow or there is none. */
    template <typename value_type_>
    value_type_* allocate(std::size_t count) noexcept {
        return std::visit(
            [&]<typename arena_type_>(arena_type_& typed) noexcept -> value_type_* {
                if constexpr (std::is_same_v<arena_type_, std::monostate>) return nullptr;
                else return typed.template allocate<value_type_>(count);
            },
            arena);
    }

    /** Whether @p address lies in memory handed out since the last @ref clear. */
    bool contains(void const* address) const noexcept {
        return std::visit(
            [&]<typename arena_type_>(arena_type_ const& typed) noexcept {
                if constexpr (std::is_same_v<arena_type_, std::monostate>) return false;
                else return typed.contains(address);
            },
            arena);
    }

    /** Rewinds the arena, so the previous call's outputs die. */
    void clear() noexcept {
        std::visit(
            []<typename arena_type_>(arena_type_& typed) noexcept {
                if constexpr (!std::is_same_v<arena_type_, std::monostate>) typed.clear();
            },
            arena);
    }

    /** Queues a copy of @p bytes of host memory at @p source to @p destination in the arena on its
     *  executor's first stream; @p source must stay alive until @ref drain. */
    status_t upload(void* destination, void const* source, std::size_t bytes) noexcept {
        return std::visit(
            [&]<typename arena_type_>(arena_type_& typed) noexcept {
                if constexpr (std::is_same_v<arena_type_, std::monostate>) return status_t::device_error_k;
                else return typed.upload(destination, source, bytes);
            },
            arena);
    }

    /** Waits once for every upload queued since the last @ref clear; none is ever queued without a
     *  device. */
    status_t drain() noexcept {
        return std::visit(
            []<typename arena_type_>(arena_type_& typed) noexcept {
                if constexpr (std::is_same_v<arena_type_, std::monostate>) return status_t::success_k;
                else return typed.drain();
            },
            arena);
    }
};

} // namespace unum::usearch::utilities
