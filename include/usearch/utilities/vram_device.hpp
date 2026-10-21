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
#include <cstdio>  // `std::snprintf` - a kernel symbol's digits
#include <cstring> // `std::strlen` - a name's length prefix in a kernel symbol

#include <mutex>       // `std::call_once` - the listing is taken once per process
#include <span>        // `std::span` - the listing handed out
#include <type_traits> // `std::is_enum_v`, `std::is_same_v` - a kernel symbol's arguments
#include <variant>     // `std::variant`, `std::monostate` - one alternative per vendor

#include <numkong/types.h> // `nk_dtype_t` - a kernel symbol's dtype argument

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
 *  distinct device over their vendor ordinals, which every shard there shares, its scratch capped
 *  at @p scratch_limit bytes per stream, zero for no ceiling: @c config_mismatch_k when they span
 *  vendors, @c device_error_k when none is named or their runtime refuses. */
inline expected<gpu_executor_t> spawn_gpu_executor(std::span<gpu_device_t const> devices,
                                                   std::size_t scratch_limit = 0) noexcept {
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
        auto [spawned, status] = executor_type_::spawn(to_span(ordinals), scratch_limit);
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
    status_t wait() noexcept {
        return std::visit(
            []<typename arena_type_>(arena_type_& typed) noexcept {
                if constexpr (std::is_same_v<arena_type_, std::monostate>) return status_t::success_k;
                else return typed.wait();
            },
            arena);
    }
};

#pragma region Kernel Symbols

/** Every GPU kernel of every design and vendor, the first template argument of each kernel
 *  template - dense @c cuda_kernel and @c amd_kernel, each modality's @c cuda_kernel - and the
 *  words of Apple's exported names. A vendor exports the kernels its design launches, and sizes
 *  its tables of loaded kernels by @c count_k. */
enum class vram_kernel_t : u32_t {

    /** Walks the graph for each row of an insertion wave and selects its forward neighbors. */
    insert_k,

    /** Links the nodes an insertion wave picked back to the wave's rows. */
    back_link_k,

    /** Walks the graph again for existing rows and re-selects their neighbors. */
    refine_k,

    /** Walks the graph for each query and ranks its matches. */
    search_k,

    /** A search rescoring its candidates through the indexer heads. */
    search_indexer_k,

    /** Scores every visible row against each query, without walking the graph. */
    scan_k,

    /** A walk landing every scored pair, rather than ranked matches. */
    multiply_k,

    /** A walk folding the values of the rows it attends to, per head. */
    attend_k,

    /** Folds each walked node's forward picks and the walkers chained onto it into its row. */
    merge_back_links_k,

    /** Rebuilds the norms column from the stored vectors. */
    derive_norms_k,

    /** Moves rows' neighbor IDs between ID spaces, packing the survivors. */
    rebase_links_k,

    /** Marks the reachability audit's landmark rows reached and in the first frontier. */
    mark_landmarks_k,

    /** Expands the reachability frontier by one level. */
    reachability_sweep_k,

    /** Counts the rows no sweep reached, collecting their IDs. */
    reachability_islands_k,

    /** Counts, for every row, the reached rows that point at it. */
    reached_in_links_k,

    /** Links each unreached island back into the reached graph. */
    reconnect_island_k,

    /** Places a split's part roots, one row per part, in the first frontier. */
    partition_roots_k,

    /** Proposes every frontier row to the unplaced rows it links to, a growth round's first. */
    partition_propose_k,

    /** Counts the rows a growth round proposed to each part. */
    partition_tally_k,

    /** Places the proposed rows of every part with room for the whole round. */
    partition_accept_k,

    /** Places the lowest-ID proposed rows of every part a round overflows, up to its room. */
    partition_fill_k,

    /** Places every row no growth round reached in the smallest parts. */
    partition_islands_k,

    /** Marks the rows linking into another part. */
    partition_seams_k,

    /** Moves every row's neighbors into its part's receiver, rebased through the slots. */
    split_links_k,

    /** Moves every row's ragged value into its part's receiver. */
    split_tape_k,

    /** Moves every fixed-width row into its part's receiver. */
    split_rows_k,

    /** Merges the attended rows of several shards' walks by the flash rule. */
    merge_attention_k,

    /** Ranks a walk's matches into the queries' rankings. */
    resolve_matches_k,

    /** Merges the rankings of several shards' walks per query. */
    merge_matches_k,

    /** Casts rows into the stored scalar. */
    cast_k,

    /** Casts rows into the stored scalar, scaled to unit RMS. */
    rmscast_k,

    /** The number of kernels, sizing the tables of loaded kernels. */
    count_k,
};

/** The Itanium spelling of @ref vram_kernel_t, a sibling of every kernel's scope, as
 *  @c vram_kernel_symbol documents. */
constexpr char const* itanium_name(vram_kernel_t) noexcept { return "NS0_9utilities13vram_kernel_tE"; }

/** Bytes of the longest symbol any kernel's spelling reaches, its terminator included. */
inline constexpr std::size_t vram_kernel_symbol_capacity_k = 256;

/** One kernel instantiation's symbol, as a module lookup takes it. */
struct vram_kernel_symbol_t {
    char text[vram_kernel_symbol_capacity_k] = {};
    std::size_t length = 0;

    /** Appends @p bytes of @p source, clipping at the capacity, which no kernel reaches. */
    void append(char const* source, std::size_t bytes) noexcept {
        for (std::size_t offset = 0; offset != bytes && length + 1 < vram_kernel_symbol_capacity_k; ++offset)
            text[length++] = source[offset];
        text[length] = 0;
    }
    void append(char const* source) noexcept { append(source, std::strlen(source)); }

    /** Appends @p name after its decimal length, as the mangling prefixes every source name. */
    void append_name(char const* name) noexcept {
        char digits[24];
        int const written = std::snprintf(digits, sizeof(digits), "%zu", std::strlen(name));
        append(digits, static_cast<std::size_t>(written));
        append(name);
    }

    /** Appends the decimal digits of @p value. */
    void append_value(unsigned long long value) noexcept {
        char digits[24];
        int const written = std::snprintf(digits, sizeof(digits), "%llu", value);
        append(digits, static_cast<std::size_t>(written));
    }
};

/** The one-letter builtin spelling of @c void or the unsigned integer @p value_type_ - @c v, then
 *  @c j, @c m or @c y, as the platform's widths name them, so @c u64_t spells as @c m on Linux and
 *  @c y elsewhere. */
template <typename value_type_>
constexpr char itanium_builtin() noexcept {
    if constexpr (std::is_void_v<value_type_>) return 'v';
    else if constexpr (std::is_same_v<value_type_, unsigned int>) return 'j';
    else if constexpr (std::is_same_v<value_type_, unsigned long>) return 'm';
    else {
        static_assert(std::is_same_v<value_type_, unsigned long long>, "Kernels take unsigned integers or void");
        return 'y';
    }
}

/** Appends one template argument: a type, an integer, @c nk_dtype_t, or an enum spelled by its own
 *  @c itanium_name overload - through @c unum::usearch, @c S0_, for @ref vram_kernel_t, or whole
 *  for a C enum. */
template <typename argument_type_>
void append_itanium_argument(vram_kernel_symbol_t& symbol, argument_type_ argument) noexcept {
    if constexpr (std::is_same_v<argument_type_, nk_dtype_t>) {
        symbol.append("L");
        symbol.append_name("nk_dtype_t");
        symbol.append_value(static_cast<unsigned long long>(argument));
        symbol.append("E");
    }
    else if constexpr (std::is_enum_v<argument_type_>) {
        symbol.append("L");
        symbol.append(itanium_name(argument));
        symbol.append_value(
            static_cast<unsigned long long>(static_cast<std::underlying_type_t<argument_type_>>(argument)));
        symbol.append("E");
    }
    else {
        char const builtin[2] = {itanium_builtin<argument_type_>(), 0};
        symbol.append("L");
        symbol.append(builtin);
        symbol.append_value(static_cast<unsigned long long>(argument));
        symbol.append("E");
    }
}

/** The tag a type argument travels as, beside the values. */
template <typename value_type_>
struct itanium_type {};

template <typename value_type_>
void append_itanium_argument(vram_kernel_symbol_t& symbol, itanium_type<value_type_>) noexcept {
    char const builtin[2] = {itanium_builtin<value_type_>(), 0};
    symbol.append(builtin);
}

/** Spells @c uuid_t, @c S3_ being @c unum::usearch::utilities, which a kernel's leading
 *  @ref vram_kernel_t argument introduces. */
inline void append_itanium_argument(vram_kernel_symbol_t& symbol, itanium_type<uuid_t>) noexcept {
    symbol.append("NS3_6uuid_tE");
}

/** Where the kernel's parameter struct lives: the kernel's own scope, or @c basic, whose arguments
 *  every modality kernel takes. */
enum class itanium_scope_t { kernel_k, basic_k };

/** The kernel's one parameter: a struct of @c scope, or a template of it instantiated with the
 *  kernel's own template argument at @c argument, counted from zero. */
struct itanium_parameter_t {
    char const* name;
    int argument = -1;
    itanium_scope_t scope = itanium_scope_t::kernel_k;

    itanium_parameter_t(char const* struct_name) noexcept : name(struct_name) {}
    itanium_parameter_t(char const* template_name, int template_argument,
                        itanium_scope_t template_scope = itanium_scope_t::kernel_k) noexcept
        : name(template_name), argument(template_argument), scope(template_scope) {}
};

/**
 *  @brief Spells the symbol of @b unum::usearch::<scope>::<kernel><arguments...>(<parameter>), the
 *      kernel's one parameter being a struct, or a struct template, of that same scope or of
 *      @c basic.
 *
 *  The substitutions are fixed by that shape: @c S_ is @c unum, @c S0_ is @c unum::usearch, @c S1_
 *  is the scope and @c S2_ the kernel's template. A @ref vram_kernel_t argument spells as
 *  @c NS0_9utilities13vram_kernel_tE, adding @c S3_ and @c S4_, of which only a @c uuid_t reuses
 *  @c S3_. The parameter spells as @c NS1_<length><parameter>E for the scope's own struct and as
 *  @c NS0_5basic<length><parameter>E for basic's - with @c I<T>E before the closing @c E when it
 *  reuses the kernel's template argument, @c T_ for the first and @c T<n-1>_ for the n-th. So
 *  @c sequence::cuda_kernel<vram_kernel_t::search_k, sz_cap_cuda_k, sz_levenshtein_bytes_k, 2,
 *  u32_t, 4> over @c basic::vram_arguments<u32_t> spells, broken here at its components:
 *  @c _ZN4unum7usearch8sequence11cuda_kernel @c I @c LNS0_9utilities13vram_kernel_tE3E
 *  @c Lm281474976710656E @c L23sz_levenshtein_symbol_t0E @c Lj2E @c j @c Lj4E @c EEv
 *  @c NS0_5basic14vram_argumentsIT3_EE
 */
template <typename... arguments_types_>
void vram_kernel_symbol(vram_kernel_symbol_t& symbol, char const* scope, char const* kernel,
                        itanium_parameter_t parameter, arguments_types_... arguments) noexcept {
    symbol = {};
    symbol.append("_ZN4unum7usearch");
    symbol.append_name(scope);
    symbol.append_name(kernel);
    symbol.append("I");
    (append_itanium_argument(symbol, arguments), ...);
    symbol.append(parameter.scope == itanium_scope_t::basic_k ? "EEvNS0_5basic" : "EEvNS1_");
    symbol.append_name(parameter.name);
    if (parameter.argument == 0) symbol.append("IT_E");
    else if (parameter.argument > 0) {
        symbol.append("IT");
        symbol.append_value(static_cast<unsigned long long>(parameter.argument - 1));
        symbol.append("_E");
    }
    symbol.append("E");
}

#pragma endregion Kernel Symbols

} // namespace unum::usearch::utilities
