/**
 *  @file include/usearch/utilities/vram_device.hpp
 *  @author Ash Vardanian
 *  @date July 22, 2026
 *  @brief The vendor-neutral face of the GPU residency: every vendor's driver, the device listing,
 *      one executor and one arena over every vendor.
 *
 *  One binary drives NVIDIA, AMD and Apple GPUs at once, each vendor through a runtime its caller
 *  loads once and declared without its SDK, so every unit names all three: NVIDIA devices are
 *  listed first, then AMD, then Apple. A vendor whose runtime is absent contributes no devices.
 *
 *  Every vendor's device header provides the same surface: a vendor-prefixed listing into a
 *  caller's buffer, an executor with a static @c spawn over its driver and vendor ordinals whose
 *  streams satisfy @c a_gpu_stream, and a device arena, made from the executor's first stream,
 *  with @c allocate, @c contains, @c clear, @c upload and @c wait. The driver tables, the stream
 *  owners, the executors, the concepts and the spawn below sit in the Shared VRAM regions, which
 *  UForm and UStore carry word for word.
 */
#pragma once

#include <cstddef> // `std::size_t`
#include <cstdint> // `std::uintptr_t` - a stream's addresses
#include <cstdio>  // `std::snprintf` - a kernel symbol's digits
#include <cstring> // `std::strlen` - a name's length prefix in a kernel symbol

#include <bit>         // `std::bit_floor` - a checkpoint batch's rows
#include <concepts>    // `std::same_as`, `std::convertible_to` - the stream and executor concepts
#include <span>        // `std::span` - the listing written
#include <type_traits> // `std::is_enum_v`, `std::is_same_v` - a kernel symbol's arguments
#include <variant>     // `std::variant`, `std::monostate` - one alternative per vendor

#include <numkong/types.h> // `nk_dtype_t` - a kernel symbol's dtype argument

#include "usearch/utilities/types.hpp"              // `gpu_device_t`
#include "usearch/utilities/vram_amd_device.hpp"    // `list_amd_gpu_devices`, `hip_executor_t`
#include "usearch/utilities/vram_apple_device.hpp"  // `list_apple_gpu_devices`, `metal_executor_t`
#include "usearch/utilities/vram_nvidia_device.hpp" // `list_nvidia_gpu_devices`, `cuda_executor_t`

namespace unum::usearch::utilities {

/** USearch's optional driver entries beyond the shared tables: virtual memory, the Tensor Memory
 *  Accelerator's encoder and cuFile on NVIDIA, and virtual memory on AMD; each resolves apart, so
 *  a missing one never empties the shared tables. */
struct gpu_driver_extensions_t {
    cuda_extensions_t cuda;
    hip_extensions_t hip;

    gpu_driver_extensions_t() noexcept : cuda(cuda_extensions_t::load()), hip(hip_extensions_t::load()) {}
};

#pragma region Shared VRAM

/** Every vendor's driver table, Apple's device contexts and the engine's optional entries, which a
 *  caller builds once per process and hands to every GPU call; a vendor whose runtime is absent
 *  keeps an empty table. */
struct gpu_drivers_t {
    cuda_api_t cuda;
    hip_api_t hip;
    metal_api_t metal;
    metal_contexts_t metal_contexts;

    /** The engine's optional entries, resolved apart from the tables above. */
    gpu_driver_extensions_t extensions;

    gpu_drivers_t() noexcept
        : cuda(cuda_api_t::load()), hip(hip_api_t::load()), metal(metal_api_t::load()), metal_contexts(metal) {}
    gpu_drivers_t(gpu_drivers_t const&) = delete;
    gpu_drivers_t& operator=(gpu_drivers_t const&) = delete;
};

/** The driver each vendor's executor spawns over: its runtime's table, or Apple's contexts. */
inline cuda_api_t const& driver_of(gpu_drivers_t& drivers, std::type_identity<cuda_executor_t>) noexcept {
    return drivers.cuda;
}
inline hip_api_t const& driver_of(gpu_drivers_t& drivers, std::type_identity<hip_executor_t>) noexcept {
    return drivers.hip;
}
inline metal_contexts_t& driver_of(gpu_drivers_t& drivers, std::type_identity<metal_executor_t>) noexcept {
    return drivers.metal_contexts;
}

/** Writes into @p into every device @p drivers reach, NVIDIA first, then AMD, then Apple, as each
 *  driver reports it, and answers how many it wrote. */
inline std::size_t list_gpu_devices(gpu_drivers_t const& drivers, std::span<gpu_device_t> into) noexcept {
    std::size_t count = list_nvidia_gpu_devices(drivers.cuda, into);
    count += list_amd_gpu_devices(drivers.hip, into.subspan(count));
    count += list_apple_gpu_devices(drivers.metal_contexts, into.subspan(count));
    return count;
}

/** What one vendor's stream answers: its device, binding it, waiting for everything queued, and
 *  the memory operations queued in its order, each with that device bound. */
template <typename stream_type_>
concept a_gpu_stream = requires(stream_type_& stream, stream_type_ const& spawned, std::size_t bytes,
                                std::uintptr_t address, bytes_span_t target) {
    { spawned.ordinal } -> std::convertible_to<std::size_t>;
    { spawned.bind() } -> std::same_as<status_t>;
    { stream.wait() } -> std::same_as<status_t>;
    { stream.allocate_async(bytes) } -> std::same_as<expected<std::uintptr_t>>;
    { stream.free_async(address) };
    { stream.copy_async(address, address, bytes) } -> std::same_as<status_t>;
    { stream.upload_async(address, bytes_view_t {target}) } -> std::same_as<status_t>;
    { stream.download_async(target, address) } -> std::same_as<status_t>;
    { stream.fill_async(address, byte_t {}, bytes) } -> std::same_as<status_t>;
};

/** What every vendor's executor answers: its vendor, and its streams, one per device in the order
 *  the caller named them, and the wait for all of them. */
template <typename executor_type_>
concept a_gpu_executor = a_gpu_stream<typename executor_type_::stream_t> &&
                         requires(executor_type_& executor, executor_type_ const& spawned, std::size_t index) {
                             { executor_type_::vendor_k } -> std::convertible_to<gpu_vendor_t>;
                             { spawned.size() } -> std::same_as<std::size_t>;
                             { executor.stream(index) } -> std::same_as<typename executor_type_::stream_t&>;
                             { executor.wait() } -> std::same_as<status_t>;
                         };

static_assert(a_gpu_executor<cuda_executor_t>);
static_assert(a_gpu_executor<hip_executor_t>);
static_assert(a_gpu_executor<metal_executor_t>);

/** Whichever vendor's executor a VRAM call runs on, or none. */
using gpu_executor_t = std::variant<std::monostate, cuda_executor_t, hip_executor_t, metal_executor_t>;

/**
 *  @brief Spawns the executor of the one vendor every device in @p devices belongs to, over that
 *      vendor's driver in @p drivers, opening each stream's memory and kernels with
 *      @p memory_arguments.
 *
 *  One stream per distinct device over their vendor ordinals, which every shard there shares: the
 *  executor opens it, or adopts the entry of @p streams beside the device, a refused spawn leaving
 *  every one of them the caller's.
 *
 *  @par Statuses:
 *  - @c invalid_argument_k No device is named, one repeats beside @p streams, or a stream is null
 *    or a default one.
 *  - @c config_mismatch_k The @p devices span vendors, or a stream sits on another device than
 *    its entry names.
 *  - @c device_k The build drives no executor of their vendor, or a device refused its stream.
 */
template <typename... memory_arguments_>
expected<gpu_executor_t> spawn_gpu_executor(gpu_drivers_t& drivers, std::span<gpu_device_t const> devices,
                                            std::span<void* const> streams,
                                            memory_arguments_&... memory_arguments) noexcept {
    if (devices.empty() || devices.size() > gpu_devices_capacity_k)
        return {{}, {status_code_t::invalid_argument_k, "An executor names one to a listing's devices"}};
    int ordinals[gpu_devices_capacity_k] = {};
    std::size_t ordinals_count = 0;
    for (gpu_device_t const& device : devices) {
        if (device.vendor != devices[0].vendor)
            return {{}, {status_code_t::config_mismatch_k, "One executor drives one GPU vendor"}};
        int const ordinal = static_cast<int>(device.vendor_ordinal);
        std::size_t earlier = 0;
        while (earlier != ordinals_count && ordinals[earlier] != ordinal) ++earlier;
        // Two adopted streams on one device would leave one of them unowned.
        if (earlier != ordinals_count && !streams.empty())
            return {{}, {status_code_t::invalid_argument_k, "Each adopted stream needs its own device"}};
        if (earlier == ordinals_count) ordinals[ordinals_count++] = ordinal;
    }
    gpu_executor_t executor;
    status_t spawning {status_code_t::device_k, "The build drives no executor of the vendor"};
    auto const spawn = [&]<typename executor_type_>(std::type_identity<executor_type_> vendor) noexcept {
        if (executor_type_::vendor_k != devices[0].vendor) return false;
        auto [spawned, status] = executor_type_::spawn(
            driver_of(drivers, vendor), std::span<int const> {ordinals, ordinals_count}, streams, memory_arguments...);
        if (status.succeeded()) executor.template emplace<executor_type_>(std::move(spawned));
        spawning = status;
        return true;
    };
    spawn(std::type_identity<cuda_executor_t> {}) || spawn(std::type_identity<hip_executor_t> {}) ||
        spawn(std::type_identity<metal_executor_t> {});
    return {std::move(executor), spawning};
}

#pragma endregion Shared VRAM

/** The position in @p executor of its stream on vendor ordinal @p ordinal, or its size where it
 *  drives none there: the one lookup a call pairs each shard with a stream through. */
template <typename executor_type_>
std::size_t stream_on(executor_type_ const& executor, std::size_t ordinal) noexcept {
    std::size_t stream = 0;
    while (stream != executor.size() && executor.stream(stream).ordinal != ordinal) ++stream;
    return stream;
}

/** The device arena each vendor's executor places a C-layer call's outputs in. */
template <typename executor_type_>
struct gpu_device_arena_of;
template <>
struct gpu_device_arena_of<cuda_executor_t> {
    using type = cuda_device_arena_t;
};
template <>
struct gpu_device_arena_of<hip_executor_t> {
    using type = hip_device_arena_t;
};
template <>
struct gpu_device_arena_of<metal_executor_t> {
    using type = metal_device_arena_t;
};

/** Device memory a C-layer call writes its outputs to, on the vendor and first stream of the
 *  executor it was made for, or none for host-resident calls; it serves calls on that executor
 *  alone, and dies before it. */
struct gpu_arena_t {
    std::variant<std::monostate, cuda_device_arena_t, hip_device_arena_t, metal_device_arena_t> arena;

    /** Places the arena on @p executor 's vendor and first stream; none for no GPU. */
    static gpu_arena_t on(gpu_executor_t& executor) noexcept {
        gpu_arena_t placed;
        std::visit(
            [&]<typename executor_type_>(executor_type_& typed) noexcept {
                if constexpr (!std::is_same_v<executor_type_, std::monostate>)
                    placed.arena.template emplace<typename gpu_device_arena_of<executor_type_>::type>(typed.stream(0));
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

    /**
     *  @brief Queues a copy of @p bytes of host memory at @p source to @p destination in the arena
     *      on its executor's first stream; @p source must stay alive until @ref wait.
     *
     *  @par Statuses:
     *  - @c device_k The arena has no device.
     */
    status_t upload(void* destination, void const* source, std::size_t bytes) noexcept {
        return std::visit(
            [&]<typename arena_type_>(arena_type_& typed) noexcept -> status_t {
                if constexpr (std::is_same_v<arena_type_, std::monostate>) return status_code_t::device_k;
                else return typed.upload(destination, source, bytes);
            },
            arena);
    }

    /** Waits once for every upload queued since the last @ref clear; none is ever queued without a
     *  device. */
    status_t wait() noexcept {
        return std::visit(
            []<typename arena_type_>(arena_type_& typed) noexcept -> status_t {
                if constexpr (std::is_same_v<arena_type_, std::monostate>) return status_code_t::success_k;
                else return typed.wait();
            },
            arena);
    }
};

/** One row a gather copies, on any vendor: the node whose stored row it reads, and the row of the
 *  outputs it writes. */
struct vram_gathered_row_t {
    u32_t node = 0;
    u32_t output_row = 0;
};
static_assert(sizeof(vram_gathered_row_t) == 8, "A gather lists its rows in eight bytes each");

/** How the last load from a path read its parts onto the device: none yet, through host memory,
 *  or every part straight onto the device through cuFile. */
enum class vram_load_path_t : u8_t { none_k, host_copy_k, cufile_k };

#pragma region Kernel Symbols

/** The GPU kernels basic graphs and every design launch, the first template argument of each
 *  shared kernel template - dense @c cuda_kernel and @c amd_kernel, each modality's
 *  @c cuda_kernel - and the words of Apple's exported names. A design's own kernels take its own
 *  enum, such as @c dense::vram_dense_kernel_t. A vendor sizes its tables of loaded kernels by
 *  @c count_k. */
enum class vram_kernel_t : u32_t {

    // Graph Building
    /** Walks the graph for each entry of an insertion wave and selects its forward neighbors. */
    insert_k,

    /** Links the nodes an insertion wave picked back to the wave's nodes. */
    back_link_k,

    /** Walks the graph again for existing nodes and re-selects their neighbors. */
    refine_k,

    /** Walks the graph for each query and ranks its matches. */
    search_k,

    /** Walks the graph for each query by its sketch and writes its nearest nodes as its search's
     *  seeds. */
    seed_k,

    /** Scores every published node for each query, a range of nodes per walker, and ranks its
     *  matches exactly. */
    scan_k,

    /** Moves nodes' neighbor IDs between ID spaces, packing the survivors. */
    rebase_links_k,

    // Rounds

    /** Files every entry of a gather's runs under its nearest hub. */
    gather_neighborhoods_k,

    /** Re-stitches each neighborhood: all against all, then each entry's diverse links. */
    restitch_k,

    // Health Check

    /** Expands one depth of a health check from its root, marking ways to the root. */
    health_expand_k,

    /** Marks the reached entries that now link to an entry reaching the root. */
    health_recheck_k,

    /** Writes the entries outside the root's component in ascending order. */
    order_islands_k,

    /** Marks the islands linking into the root's component and their ways to the root. */
    health_islands_k,

    /** Propagates the least rank along the islands' links, both ways, naming their components. */
    island_components_k,

    /** Counts unlinked and underlinked entries and files each island's parent. */
    health_count_links_k,

    /** Labels the islands' roles and entries and counts the weak components. */
    label_islands_k,

    /** Links every island component through the root's component, with no distance. */
    heal_islands_k,

    // Reachability Audit
    /** Marks the reachability audit's landmark nodes reached and in the first frontier. */
    mark_landmarks_k,

    /** Expands the reachability frontier by one level. */
    reachability_sweep_k,

    /** Marks the nodes reaching back to the landmarks through a node that did a level earlier. */
    reachability_back_sweep_k,

    /** Counts the nodes no sweep reached, or that reach no landmark back, collecting their IDs. */
    reachability_islands_k,

    /** Counts, for every node, the reached nodes that point at it. */
    reached_in_links_k,

    /** Links each island into the graph the audit reached both ways. */
    reconnect_island_k,

    // Partitioning
    /** Places a split's part roots, one node per part, in the first frontier. */
    partition_roots_k,

    /** Proposes every frontier node to the unplaced nodes it links to, a growth round's first. */
    partition_propose_k,

    /** Counts the nodes a growth round proposed to each part. */
    partition_count_k,

    /** Places the proposed nodes of every part with room for the whole round. */
    partition_accept_k,

    /** Places the lowest-ID proposed nodes of every part a round overflows, up to its room. */
    partition_fill_k,

    /** Places every node no growth round reached in the smallest parts. */
    partition_islands_k,

    /** Marks the nodes linking into another part. */
    partition_picked_k,

    // Split Moves

    /** Moves every node's neighbors into its part's target, rebased through the slots. */
    split_links_k,

    /** Moves every node's ragged value into its part's target. */
    split_tape_k,

    /** Moves every node's vector, norms and key into its part's target. */
    split_nodes_k,

    // Gathers and Products

    /** Copies listed nodes' rows into a caller's matrix, cast into its scalar. */
    gather_vectors_k,

    /** Sums a batch's per-query counts into one word. */
    sum_counts_k,

    /** Resolves one shard's walked products into keys through its device key column. */
    resolve_products_k,

    /** Packs every query's kept products back to back from its offset. */
    compact_products_k,

    /** Copies listed nodes' ragged values back to back into a caller's tape. */
    gather_tape_k,

    // Exact MaxSim

    /** Collects every query's distinct keys across its token rows and the shards. */
    nominate_keys_k,

    /** Takes one shard's best dot of every nominee per token row of its query. */
    maxsim_rescore_k,

    /** Takes the best of every shard's dots per token row, summed over the rows. */
    maxsim_fold_k,

    /** Appends every query's best-scoring nominees to a tape. */
    top_k_k,

    // Radius Walks

    /** Counts each query's entries within its radius. */
    within_count_k,

    /** Writes each query's entries within its radius from its scanned offset. */
    within_write_k,

    // Loads

    /** Widens a load's saved neighbors into padded rows of a basic graph, checking every ID. */
    widen_links_k,

    /** Stores a load's saved values into a basic graph's tape where a kit converts them. */
    load_values_k,

    /** Fills the sketch in front of every sequence of an insert's staged values. */
    sketch_values_k,

    /** The number of kernels, sizing the tables of loaded kernels. */
    count_k,
};

/** The role @p kernel scores in: a search for the walk ranking a query's matches, a build for the
 *  walks, merges and rounds that link the graph. */
constexpr score_role_t vram_kernel_role(vram_kernel_t kernel) noexcept {
    return kernel == vram_kernel_t::search_k || kernel == vram_kernel_t::scan_k ? score_role_t::search_k
                                                                                : score_role_t::build_k;
}

/** What a sketch phase reads: value @c v of the batch spans
 *  @b [offsets[v]+shift,offsets[v+1]+shift) of @c values, and @c tables is the kit's own. An
 *  insert's stored values take their sketches of @c sketch_bytes in front of them; a search's bare
 *  queries take them in rows of @c sketches, a sketch apart. */
struct vram_sketch_t {
    byte_t* values = nullptr;
    u64_t const* offsets = nullptr;
    u64_t shift = 0;
    u32_t count = 0;
    u32_t sketch_bytes = 0;
    byte_t const* tables = nullptr;
    byte_t* sketches = nullptr;
};

/** The Itanium spelling of @ref unum::usearch::utilities::vram_kernel_t, a sibling of every
 *  kernel's scope, as @c vram_kernel_symbol documents. */
constexpr char const* kernel_enum_mangling(vram_kernel_t) noexcept { return "NS0_9utilities13vram_kernel_tE"; }

/** Bytes of the longest symbol any kernel's spelling reaches, its terminator included. */
inline constexpr std::size_t vram_kernel_symbol_capacity_k = 256;

/** Nodes, as a power of two, one half of a save's or a load's bounce moves at most, so its two
 *  halves pin a few hundred megabytes at the widest nodes. */
inline constexpr u32_t crossing_nodes_log2_k = 19;

/** Bytes one checkpoint batch pins at most: its four driver calls, tens of microseconds together,
 *  stay under 1% of the 5 ms its transfer takes at 50 GB/s. */
inline constexpr std::size_t checkpoint_batch_bytes_k = std::size_t {256} << 20;

/** Rows, as a power of two, one staging window holds at most however narrow they are: a
 *  checkpoint batch, a half of the key staging and a batch verb's window on one shard, so a half
 *  of 8-byte keys pins 8 MiB. */
inline constexpr u32_t window_rows_log2_k = 20;

/** Rows one checkpoint batch of rows @p row_bytes wide moves: a power of two within
 *  @c checkpoint_batch_bytes_k, at least one and at most a window. */
constexpr std::size_t checkpoint_batch_rows(std::size_t row_bytes) noexcept {
    std::size_t const fitting = std::bit_floor(checkpoint_batch_bytes_k / max_of<std::size_t>(row_bytes, 1));
    return min_of(max_of<std::size_t>(fitting, 1), power_of_two<std::size_t>(window_rows_log2_k));
}

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

/** Appends one template argument: a type, an integer, @c nk_dtype_t, or an enum spelled by the
 *  @c kernel_enum_mangling overload beside it, found by its namespace - through @c unum::usearch,
 *  @c S0_, for @ref unum::usearch::utilities::vram_kernel_t, through the kernel's scope, @c S1_,
 *  for a design's own enum, or whole for a C enum. */
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
        symbol.append(kernel_enum_mangling(argument));
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

/** Spells @c uuid128_t, @c S3_ being @c unum::usearch::utilities, which a kernel's leading
 *  @ref unum::usearch::utilities::vram_kernel_t argument introduces. */
inline void append_itanium_argument(vram_kernel_symbol_t& symbol, itanium_type<uuid128_t>) noexcept {
    symbol.append("NS3_9uuid128_tE");
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
 *  @brief Spells the symbol of a kernel by its scope, name, template arguments and parameter, the
 *      kernel's one parameter being a struct, or a struct template, of that same scope or of
 *      @c basic.
 *
 *  The substitutions are fixed by that shape: @c S_ is @c unum, @c S0_ is @c unum::usearch, @c S1_
 *  is the scope and @c S2_ the kernel's template. A @ref unum::usearch::utilities::vram_kernel_t
 *  argument spells as @c NS0_9utilities13vram_kernel_tE, adding @c S3_ and @c S4_, of which only a
 *  @c uuid128_t reuses @c S3_. The parameter spells as @c NS1_ , the name's length, the name and
 *  @c E for the scope's own struct and as @c NS0_5basic , the name's length, the name and @c E for
 *  basic's - with @c I<T>E before the closing @c E when it reuses the kernel's template argument,
 *  @c T_ for the first and @c T<n-1>_ for the n-th. So
 *  @c sequence::cuda_kernel<vram_kernel_t::insert_k, u8_ampere_banded_overlap_kit_t, 1, u32_t, 0,
 *  4> over @c basic::vram_arguments<u32_t> spells, broken here at its components:
 *  @c _ZN4unum7usearch8sequence11cuda_kernel @c I @c LNS0_9utilities13vram_kernel_tE0E
 *  @c NS1_30u8_ampere_banded_overlap_kit_tE @c Lj1E @c j @c Lj0E @c Lj4E @c EEv
 *  @c NS0_5basic14vram_argumentsIT2_EE
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
