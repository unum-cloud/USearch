/**
 *  @brief Vector Search index for pre-constructed read-only graphs on block storage.
 *  @author Ash Vardanian
 *  @file include/usearch/dense/nvme_index.hpp
 *  @date July 22, 2026
 *
 *  Unlike the mutation-oriented in-memory design, this header is built around one idea:
 *  a single graph traversal is a chain of dependent reads, each bounded by device latency,
 *  so throughput comes from keeping many independent traversals in flight. The traversal
 *  logic is therefore a pure state machine - it never performs I/O itself, only emits
 *  read requests and consumes completions - and pluggable per-OS engines pump those
 *  requests through whatever asynchronous interface the platform offers.
 *
 *  @section dense_nvme_engines I/O Engines
 *
 *  Seven engines satisfy the same @c dense_io_engine concept, each a standalone type
 *  named after the exact platform capability it drives, selected at compile time and
 *  instantiated once per traversal context:
 *
 *  - @c dense_posix_sync_engine - blocking @c pread, the portable reference & macOS path,
 *  - @c dense_linux_aio_engine - @c io_submit / @c io_getevents, for io_uring-gated clouds,
 *  - @c dense_linux_uring_read_engine - io_uring @c IORING_OP_READ on any descriptor,
 *  - @c dense_linux_uring_fixed_engine - @c IORING_OP_READ_FIXED over registered buffers & files,
 *  - @c dense_linux_uring_polled_engine - registered resources + @c IORING_SETUP_IOPOLL,
 *  - @c dense_linux_uring_nvme_engine - NVMe passthrough via @c IORING_OP_URING_CMD,
 *  - @c dense_windows_ioring_engine - I/O Rings on Windows 11 22H2 & newer.
 *
 *  Engines own the aligned staging-buffer arena - the natural consequence of buffer
 *  registration, which is an engine-side resource on every ring API - and the state
 *  machine addresses buffers only by index. Requests carry an offset, a length, a
 *  buffer index, and an opaque tag; completions return the tag and a signed result.
 *  No pointers cross the contract in either direction. Building the Linux ring engines
 *  assumes kernel headers 6.1 or newer - the deferred-task-run era every supported
 *  distro ships - while runtime degrades gracefully on older kernels through setup
 *  fallbacks and failed-registration probes.
 *
 *  @section dense_nvme_alignment Platform Alignment
 *
 *  The engine contract is the intersection of four asynchronous-read APIs - one
 *  bullet per contract verb, platforms in a fixed order: io_uring, then Linux AIO,
 *  then Windows IoRing, then the portable & macOS path:
 *
 *  - @b Create - @c io_uring_setup plus the ring mappings; @c io_setup for AIO;
 *      @c CreateIoRing on Windows; nothing to create portably.
 *  - @b Encode - an SQE written in the shared ring; an @c iocb in a caller array;
 *      @c BuildIoRingReadFile builds into the ring; portably a staged request.
 *  - @b Submit - one @c io_uring_enter per batch; @c io_submit; @c SubmitIoRing;
 *      portably a loop of @c pread calls.
 *  - @b Reap - the CQ ring drained in user space; @c io_getevents;
 *      @c PopIoRingCompletion, also syscall-free; portably inline with submission.
 *  - @b Register @b buffers - @c IORING_REGISTER_BUFFERS;
 *      no AIO equivalent; @c BuildIoRingRegisterBuffers; none portably.
 *  - @b Register @b files - @c IORING_REGISTER_FILES;
 *      no AIO equivalent; @c BuildIoRingRegisterFileHandles; none portably.
 *  - @b Cookie - @c user_data on the SQE & CQE; @c aio_data echoed in
 *      @c io_event.data; IoRing's @c UserData; portably staged alongside.
 *  - @b Probe - setup-flag @c EINVAL fallback chains; AIO is assumed present;
 *      @c IsIoRingOpSupported; the portable path is the fallback itself.
 *
 *  The SQE fields a read actually consumes: @c opcode, @c flags, @c fd, @c off,
 *  @c addr, @c len, @c user_data - plus @c buf_index for @c READ_FIXED, where @c fd
 *  becomes a registered-file index under @c IOSQE_FIXED_FILE - plus @c cmd_op and the
 *  72-byte inline @c nvme_uring_cmd for passthrough. A CQE returns only @c user_data,
 *  @c res & @c flags. Everything else in those 64/128-byte slots stays zero, which is
 *  why the request POD below carries exactly four words.
 *
 *  @section dense_nvme_format Storage Format
 *
 *  The payload opens with the shared @c dense_index_header_t, then a plain array of
 *  @b [node_count+1] 64-bit byte offsets, then the node records. Each record is fully
 *  self-contained - neighbor ids, pre-computed norms, and the vector - padded to the
 *  512-byte transfer block, so one read services one hop of one traversal. Neighbor
 *  lists are variable-length: the record stores its own count, and the offsets array
 *  is the only random-access directory.
 *
 *  @section dense_nvme_concurrency Concurrency
 *
 *  The graph is immutable, so the index object holds no atomics, no locks, and no
 *  compare-and-swap loops. All mutable state - slots, queues, the engine with its
 *  arena - lives in the per-thread @c dense_nvme_context, and no two threads
 *  ever share one. The only spinning in the design is the engine-side completion
 *  wait, delegated to the same ForkUnion monitor-waiter vocabulary the in-memory
 *  index uses for lock sentinels: @c TPAUSE on x86, @c WFET on Arm, plain yields
 *  elsewhere - energy-efficient busy-waiting instead of hard spins.
 *
 *  @section dense_nvme_rnd R&D Emulation
 *
 *  Development against emulated NVMe latency profiles uses the in-kernel @c null_blk
 *  driver; a RAM-rich box models any device tier without hardware:
 *
 *  @code{.sh}
 *  modprobe null_blk nr_devices=0
 *  mkdir /sys/kernel/config/nullb/nvme0 && cd /sys/kernel/config/nullb/nvme0
 *  echo 512   > blocksize        # 512e; use 4096 to model 4Kn formats
 *  echo 4096  > size             # MB
 *  echo 1     > memory_backed
 *  echo 2     > irqmode          # hrtimer completions
 *  echo 80000 > completion_nsec  # ~80us enterprise-TLC read
 *  echo 8     > submit_queues; echo 1024 > hw_queue_depth
 *  echo 2     > poll_queues      # for the polled engine & fio --hipri
 *  echo 1     > power            # creates /dev/nvme0 (named after this directory!)
 *  @endcode
 *
 *  Two caveats measured on 6.11: the block device is named after the configfs
 *  directory, not @c nullb0; and @c completion_nsec is honored accurately on IRQ
 *  queues but only coarsely on poll queues, so calibrate polled-path latency
 *  against an IRQ-mode run of the same instance.
 */
#pragma once

#if defined(__linux__)
#include <fcntl.h>         // `open`, `O_DIRECT`
#include <linux/aio_abi.h> // `aio_context_t`, `struct iocb`, `IOCB_CMD_PREAD`
#include <linux/io_uring.h>
#include <linux/nvme_ioctl.h> // `struct nvme_uring_cmd`, `NVME_URING_CMD_IO`
#include <sys/ioctl.h>        // `NVME_IOCTL_ID`
#include <sys/mman.h>         // ring mappings
#include <sys/syscall.h>      // `SYS_io_uring_setup` & friends
#include <sys/uio.h>          // `struct iovec` for buffer registration
#include <unistd.h>           // `pread`, `close`, `syscall`
#elif defined(__APPLE__)
#include <fcntl.h>  // `open`, `F_NOCACHE`
#include <unistd.h> // `pread`, `close`
#elif defined(_WIN32)
#include <ioringapi.h> // `CreateIoRing`, `BuildIoRingReadFile`
#include <windows.h>
#endif

#include <forkunion/capabilities.hpp> // `fu::preferred_yield_t` - the default waiter
#include <forkunion/types.hpp>
#include <numkong/tensor.hpp>

#include "usearch/types.hpp"
#include "usearch/ram_queue.hpp"
#include "usearch/dense/ram_metric.hpp"

namespace unum {
namespace usearch {

#pragma region Vocabulary

/** @brief Transfer granularity of the record section: every record begins and ends on
 *      a 512-byte boundary, matching the finest NVMe logical block, so direct I/O works
 *      against 512e devices and wastes at most one block on 4Kn ones. */
inline constexpr std::size_t dense_nvme_block_k = 512;

/** @brief Alignment of every staging buffer - the strictest direct-I/O memory alignment
 *      any mainstream device demands, and a page, so one buffer never splits pages. */
inline constexpr std::size_t dense_nvme_alignment_k = 4096;

/** @brief Identity of one in-flight traversal slot within a context's slot array -
 *      packed into the round-tripped tag and used to index the slots. */
using dense_traversal_slot_id_t = u32_t;

/** @brief Index into an engine's staging-buffer arena - leased by the state machine,
 *      carried by every request, and the @c buf_index of registered-buffer submissions. */
using dense_staging_buffer_id_t = u32_t;

/** @brief Feature bits an engine publishes through its @c capabilities_k constant, so
 *      callers, tests, and docs introspect variants uniformly instead of by type name. */
enum class dense_io_capability_t : u32_t {
    none_k = 0,
    /** @brief Staging buffers are pre-registered with the kernel/ring - no per-read
     *      page pinning; roughly an 11% throughput gain at NVMe rates. */
    fixed_buffers_k = 1u << 0,
    /** @brief Completions are polled from device queues instead of interrupt-driven;
     *      needs poll-capable queues & direct I/O; ~21% on top of the fixed path. */
    polled_k = 1u << 1,
    /** @brief Requests address raw LBAs on a namespace, not file offsets - the
     *      filesystem & block layer are bypassed; ~20-30% over the block path. */
    lba_addressing_k = 1u << 2,
};

inline constexpr dense_io_capability_t operator|(dense_io_capability_t a, dense_io_capability_t b) noexcept {
    return static_cast<dense_io_capability_t>(static_cast<u32_t>(a) | static_cast<u32_t>(b));
}
inline constexpr bool operator&(dense_io_capability_t a, dense_io_capability_t b) noexcept {
    return (static_cast<u32_t>(a) & static_cast<u32_t>(b)) != 0;
}

/** @brief How a @c pump call treats completions: return with whatever is already
 *      reaped, or insist on at least one - the polling-vs-blocking axis shared by
 *      every engine. */
enum class dense_io_wait_t : u32_t {
    /** @brief Submit & reap what's there; never sleep. */
    nonblocking_k = 0,
    /** @brief Submit, then hold until at least one completion lands. */
    blocking_k = 1,
};

/** @brief Whether the payload descriptor routes through the page cache or around it -
 *      @c O_DIRECT on Linux, @c F_NOCACHE on macOS, @c FILE_FLAG_NO_BUFFERING on Windows. */
enum class dense_io_caching_t : u32_t {
    cached_k = 0,
    direct_k = 1,
};

/** @brief Why a record was fetched; routes the completion inside the state machine. */
enum class dense_io_purpose_t : u32_t {
    /** @brief The record's vector will be scored against the slot's query. */
    score_k = 0,
    /** @brief The record's neighbor list will extend the slot's frontier. */
    expand_k = 1,
};

/** @brief Opaque per-read cookie: the issuing slot, the staging buffer, and the purpose,
 *      packed into the one 64-bit word every ring API round-trips untouched -
 *      @c sqe->user_data, @c iocb.aio_data, or IoRing's @c UserData. */
struct dense_io_tag_t {
    dense_traversal_slot_id_t slot;
    dense_staging_buffer_id_t buffer : 31;
    u32_t purpose : 1;

    inline u64_t packed() const noexcept {
        return static_cast<u64_t>(slot) << 32 | static_cast<u64_t>(buffer) << 1 | static_cast<u64_t>(purpose);
    }
    inline static dense_io_tag_t unpack(u64_t word) noexcept {
        dense_io_tag_t tag;
        tag.slot = static_cast<u32_t>(word >> 32);
        tag.buffer = static_cast<u32_t>(word) >> 1;
        tag.purpose = static_cast<u32_t>(word) & 1u;
        return tag;
    }
};

/** @brief One read order from the state machine to an engine: where, how much, into
 *      which arena slot, and the cookie to hand back. Offsets are always bytes from
 *      the start of the payload; LBA-addressed engines rescale internally. The
 *      destination address never appears - the engine derives it from @p buffer,
 *      which is also the @c buf_index of registered-buffer submissions. */
struct dense_io_request_t {
    /** @brief Byte offset of the record within the payload; block-aligned by format. */
    u64_t offset;
    /** @brief Bytes to transfer; a whole number of @ref dense_nvme_block_k blocks. */
    u32_t length;
    /** @brief Index into the engine's staging arena - stays valid until the matching
     *      completion is consumed & the state machine recycles the index. */
    dense_staging_buffer_id_t buffer;
    /** @brief Round-tripped @ref dense_io_tag_t in packed form; opaque to engines. */
    u64_t tag;
};

/** @brief One completed read, surfaced by an engine's @c pop. A negative @c result is
 *      a negated @c errno -like error; otherwise it's the byte count transferred. */
struct dense_io_completion_t {
    u64_t tag;
    std::int64_t result;
};

/**
 *  @brief Compile-time contract every I/O engine satisfies. @c push only encodes -
 *      no syscall; @c pump submits everything encoded and reaps what completed,
 *      holding for at least one completion under @c blocking_k; @c pop drains
 *      reaped completions one at a time; @c buffer maps an arena index to the
 *      memory a completed read landed in.
 */
template <typename engine_type_>
concept dense_io_engine = requires(engine_type_& engine, dense_io_request_t request, //
                                   dense_io_completion_t& completion, dense_io_wait_t wait,
                                   dense_staging_buffer_id_t index) {
    { engine.push(request) } noexcept -> std::same_as<bool>;
    { engine.pump(wait) } noexcept -> std::same_as<status_t>;
    { engine.pop(completion) } noexcept -> std::same_as<bool>;
    { engine.buffer(index) } noexcept -> std::same_as<std::byte*>;
    { engine_type_::capabilities_k } -> std::convertible_to<dense_io_capability_t>;
};

#pragma endregion Vocabulary

#pragma region Layout

/**
 *  @brief Byte arithmetic of the on-disk format, shared by readers and future writers.
 *
 *  One record, from its block-aligned start:
 *
 *      u32_t neighbors_count;
 *      u32_t self_id;
 *      u32_t neighbor_ids[neighbors_count];
 *      ...padding to 16 bytes...
 *      vector_norms<dot_result_t> norms;
 *      ...padding to 64 bytes...
 *      scalar_t vector[dimensions];
 *      ...padding to the 512-byte block...
 *
 *  The 64-byte vector alignment keeps SIMD loads within the staging buffer natural;
 *  the trailing block padding makes every record's extent a valid direct-I/O transfer.
 */
struct dense_nvme_layout_t {

    /** @brief Offsets array begins right after the 32-byte shared header. */
    inline static constexpr std::size_t offsets_begin_k = sizeof(dense_index_header_t);

    inline static constexpr std::size_t offsets_bytes(std::size_t node_count) noexcept {
        return (node_count + 1) * sizeof(u64_t);
    }
    /** @brief First byte of the record section - the metadata rounded up to a whole block. */
    inline static constexpr std::size_t records_begin(std::size_t node_count) noexcept {
        return fu::round_up_to_multiple(offsets_begin_k + offsets_bytes(node_count), dense_nvme_block_k);
    }

    /** @brief The record's own id sits right after the count: fetched records identify
     *      themselves, so completions need no side-channel id bookkeeping. */
    inline static constexpr std::size_t self_id_offset_k = sizeof(dense_node_id_t);
    inline static constexpr std::size_t neighbor_ids_offset_k = 2 * sizeof(dense_node_id_t);
    inline static constexpr std::size_t norms_offset(std::size_t neighbors_count) noexcept {
        return fu::round_up_to_multiple(neighbor_ids_offset_k + neighbors_count * sizeof(dense_node_id_t),
                                        std::size_t(16));
    }
    inline static constexpr std::size_t vector_offset(std::size_t neighbors_count, std::size_t norms_bytes) noexcept {
        return fu::round_up_to_multiple(norms_offset(neighbors_count) + norms_bytes, std::size_t(64));
    }
    inline static constexpr std::size_t record_bytes(std::size_t neighbors_count, std::size_t norms_bytes,
                                                     std::size_t vector_bytes) noexcept {
        return fu::round_up_to_multiple(vector_offset(neighbors_count, norms_bytes) + vector_bytes, dense_nvme_block_k);
    }
};

/**
 *  @brief Decoded view over one staged record: the format arithmetic of
 *      @ref dense_nvme_layout_t in one place, every accessor a zero-copy pointer
 *      into the arena buffer DMA filled. Readers & future writers share it, so the
 *      offset choreography can never drift between them.
 */
struct dense_nvme_record_t {
    std::byte const* bytes;

    inline u32_t neighbors_count() const noexcept { return *reinterpret_cast<u32_t const*>(bytes); }
    inline dense_node_id_t self_id() const noexcept {
        return *reinterpret_cast<dense_node_id_t const*>(bytes + dense_nvme_layout_t::self_id_offset_k);
    }
    inline dense_node_id_t const* neighbor_ids() const noexcept {
        return reinterpret_cast<dense_node_id_t const*>(bytes + dense_nvme_layout_t::neighbor_ids_offset_k);
    }
    template <typename norms_type_>
    inline norms_type_ norms() const noexcept {
        return *reinterpret_cast<norms_type_ const*>( //
            bytes + dense_nvme_layout_t::norms_offset(neighbors_count()));
    }
    template <typename scalar_type_, typename norms_type_>
    inline scalar_type_ const* vector() const noexcept {
        return reinterpret_cast<scalar_type_ const*>(
            bytes + dense_nvme_layout_t::vector_offset(neighbors_count(), sizeof(norms_type_)));
    }
};

#pragma endregion Layout

#pragma region Engines

#if !defined(_WIN32)
/** @brief Closes a POSIX descriptor at scope exit, so early returns need no manual
 *      teardown - the descriptor-side sibling of the ring's self-unmapping regions. */
struct dense_posix_descriptor_t {
    int file_descriptor = -1;

    dense_posix_descriptor_t() noexcept = default;
    explicit dense_posix_descriptor_t(int descriptor) noexcept : file_descriptor(descriptor) {}
    dense_posix_descriptor_t(dense_posix_descriptor_t const&) = delete;
    dense_posix_descriptor_t& operator=(dense_posix_descriptor_t const&) = delete;
    ~dense_posix_descriptor_t() noexcept {
        if (file_descriptor >= 0) ::close(file_descriptor);
    }
};
#endif

/**
 *  @brief The aligned staging arena every engine owns: @b [buffer_count] identically
 *      sized, page-aligned record buffers, allocated once from the caller's allocator -
 *      so a NUMA-domain allocator lands the arena on the requesting socket. Engines
 *      with registered buffers hand exactly these extents to the kernel.
 */
template <typename allocator_type_>
struct dense_io_arena {
    using bytes_allocator_t = typename std::allocator_traits<allocator_type_>::template rebind_alloc<std::byte>;
    using bytes_tensor_t = nk::tensor<std::byte, bytes_allocator_t, 1>;

    bytes_tensor_t storage;
    std::size_t stride = 0;
    /** @brief Cached aligned base: the arena never moves after @c reserve, so every
     *      @c buffer lookup is one multiply-add instead of a repeated round-up. */
    std::byte* base_ = nullptr;

    inline status_t reserve(std::size_t buffer_count, std::size_t buffer_stride,
                            bytes_allocator_t const& allocator) noexcept {
        storage = bytes_tensor_t(allocator);
        stride = fu::round_up_to_multiple(buffer_stride, dense_nvme_alignment_k);
        std::size_t bytes = buffer_count * stride + dense_nvme_alignment_k;
        if (!storage.reserve(bytes)) return status_t::bad_alloc_k;
        if (!storage.try_resize({bytes})) return status_t::bad_alloc_k;
        std::uintptr_t raw = reinterpret_cast<std::uintptr_t>(storage.data());
        base_ = reinterpret_cast<std::byte*>(fu::round_up_to_multiple(raw, dense_nvme_alignment_k));
        return status_t::success_k;
    }
    inline std::byte* buffer(dense_staging_buffer_id_t index) const noexcept { return base_ + index * stride; }
};

/**
 *  @brief Fixed-capacity completion queue of the engines that have no kernel-shared
 *      completion ring - the synchronous and AIO designs. The fill is the live shape
 *      over a frozen reservation, and completions drain last-first, which the state
 *      machine welcomes: any completion order drives a valid traversal.
 */
template <typename allocator_type_>
struct dense_io_completions {
    using completions_allocator_t =
        typename std::allocator_traits<allocator_type_>::template rebind_alloc<dense_io_completion_t>;
    using completions_tensor_t = nk::tensor<dense_io_completion_t, completions_allocator_t, 1>;

    completions_tensor_t queue;

    inline status_t reserve(std::size_t queue_depth, allocator_type_ const& allocator) noexcept {
        queue = completions_tensor_t(completions_allocator_t(allocator));
        return queue.reserve(queue_depth) ? status_t::success_k : status_t::bad_alloc_k;
    }
    inline bool complete(dense_io_completion_t completion) noexcept {
        std::size_t fill = queue.extent(0);
        if (!queue.try_resize({fill + 1})) return false;
        queue[fill] = completion;
        return true;
    }
    inline bool pop(dense_io_completion_t& completion) noexcept {
        std::size_t fill = queue.extent(0);
        if (!fill) return false;
        completion = queue[fill - 1];
        return queue.try_resize({fill - 1});
    }
};

/**
 *  @brief Request & completion staging of the synchronous engine: the push-encodes /
 *      pump-submits contract forces requests to live somewhere between the two calls,
 *      and a blocking @c pread has no kernel-side queue to be that somewhere.
 */
template <typename allocator_type_>
struct dense_io_staging {
    using requests_allocator_t =
        typename std::allocator_traits<allocator_type_>::template rebind_alloc<dense_io_request_t>;
    using requests_tensor_t = nk::tensor<dense_io_request_t, requests_allocator_t, 1>;

    requests_tensor_t pending;
    dense_io_completions<allocator_type_> completions;

    inline status_t reserve(std::size_t queue_depth, allocator_type_ const& allocator) noexcept {
        pending = requests_tensor_t(requests_allocator_t(allocator));
        if (!pending.reserve(queue_depth)) return status_t::bad_alloc_k;
        return completions.reserve(queue_depth, allocator);
    }
    inline bool push(dense_io_request_t request) noexcept {
        std::size_t fill = pending.extent(0);
        if (!pending.try_resize({fill + 1})) return false;
        pending[fill] = request;
        return true;
    }
    inline bool complete(dense_io_completion_t completion) noexcept { return completions.complete(completion); }
    inline bool pop(dense_io_completion_t& completion) noexcept { return completions.pop(completion); }
};

/**
 *  @brief Portable blocking engine: @c pump services every pending request with one
 *      @c pread apiece on the calling thread. The reference implementation of the
 *      concept, the debugging harness, and the macOS production path - where a pool
 *      of caller threads, each owning one context, is the realistic ceiling anyway.
 *
 *  macOS specifics: there is no async-file syscall surface - POSIX AIO is capped at
 *  a queue depth of ~16 by @c kern.aiomax -class sysctls and kqueue cannot watch
 *  regular files - so threads x @c pread with @c F_NOCACHE is the platform idiom.
 *  Note @c F_NOCACHE is a hint, not @c O_DIRECT: unaligned I/O silently falls back
 *  to the buffered path, and resident pages are still served from cache.
 */
template <typename wait_type_ = fu::preferred_yield_t, typename allocator_type_ = std::allocator<std::byte>>
struct dense_posix_sync_engine {
    using wait_t = wait_type_;
    using allocator_t = allocator_type_;

    inline static constexpr dense_io_capability_t capabilities_k = dense_io_capability_t::none_k;

    dense_posix_sync_engine() noexcept = default;
    dense_posix_sync_engine(dense_posix_sync_engine const&) = delete;
    dense_posix_sync_engine& operator=(dense_posix_sync_engine const&) = delete;

    /** @brief Binds the engine to an open descriptor, sizes the staging arrays, and
     *      claims the buffer arena - one buffer per possible outstanding request.
     *
     *  @param[in] file_descriptor Readable descriptor of the payload file or device.
     *  @param[in] queue_depth Most requests ever outstanding between two pumps.
     *  @param[in] buffer_stride Bytes per staging buffer - the longest record.
     *  @param[in] allocator Allocator behind the arena & staging arrays.
     */
    inline status_t open(int file_descriptor, std::size_t queue_depth, std::size_t buffer_stride,
                         allocator_t const& allocator) noexcept {
        file_descriptor_ = file_descriptor;
        status_t arena_status = arena_.reserve(queue_depth, buffer_stride, allocator);
        if (arena_status != status_t::success_k) return arena_status;
        return staging_.reserve(queue_depth, allocator);
    }

    inline std::byte* buffer(dense_staging_buffer_id_t index) noexcept { return arena_.buffer(index); }
    inline bool push(dense_io_request_t request) noexcept { return staging_.push(request); }

    inline status_t pump(dense_io_wait_t wait) noexcept {
        (void)wait; // Every pending request completes synchronously, so waiting is implicit.
        for (std::size_t i = 0; i != staging_.pending.extent(0); ++i) {
            dense_io_request_t const& request = staging_.pending[i];
#if defined(_WIN32)
            std::int64_t result = -1; // The portable engine is POSIX-only; Windows uses I/O Rings.
#else
            ssize_t bytes = ::pread(file_descriptor_, arena_.buffer(request.buffer), request.length,
                                    static_cast<off_t>(request.offset));
            std::int64_t result = bytes < 0 ? -static_cast<std::int64_t>(errno) : bytes;
#endif
            if (!staging_.complete(dense_io_completion_t {request.tag, result})) return status_t::overflow_k;
        }
        return staging_.pending.try_resize({0}) ? status_t::success_k : status_t::overflow_k;
    }

    inline bool pop(dense_io_completion_t& completion) noexcept { return staging_.pop(completion); }

  private:
    int file_descriptor_ = -1;
    dense_io_arena<allocator_t> arena_;
    dense_io_staging<allocator_t> staging_;
};

#if defined(__linux__)

/**
 *  @brief Linux AIO engine over raw @c io_submit / @c io_getevents syscalls, with
 *      @c O_DIRECT descriptors. Roughly two thirds of io_uring's per-core throughput,
 *      but its syscalls predate every seccomp profile in existence - the engine for
 *      container platforms and hardened fleets where io_uring is administratively
 *      gated: default Docker/containerd seccomp profiles block the io_uring family,
 *      gVisor rejects it, and the 6.6+ @c kernel.io_uring_disabled sysctl can turn
 *      it off fleet-wide.
 *
 *  The @c iocb fields a read consumes: @c aio_fildes, @c aio_lio_opcode, @c aio_buf,
 *  @c aio_nbytes, @c aio_offset & @c aio_data; the @c io_event returns @c data & @c res.
 *  True asynchrony holds only on the @c O_DIRECT path - buffered submissions can
 *  block inside @c io_submit - which costs this design nothing, since every record
 *  read is already block-aligned.
 */
template <typename wait_type_ = fu::preferred_yield_t, typename allocator_type_ = std::allocator<std::byte>>
struct dense_linux_aio_engine {
    using wait_t = wait_type_;
    using allocator_t = allocator_type_;
    using control_blocks_allocator_t =
        typename std::allocator_traits<allocator_type_>::template rebind_alloc<struct iocb>;
    using control_pointers_allocator_t =
        typename std::allocator_traits<allocator_type_>::template rebind_alloc<struct iocb*>;
    using events_allocator_t = typename std::allocator_traits<allocator_type_>::template rebind_alloc<struct io_event>;
    using control_blocks_tensor_t = nk::tensor<struct iocb, control_blocks_allocator_t, 1>;
    using control_pointers_tensor_t = nk::tensor<struct iocb*, control_pointers_allocator_t, 1>;
    using events_tensor_t = nk::tensor<struct io_event, events_allocator_t, 1>;

    inline static constexpr dense_io_capability_t capabilities_k = dense_io_capability_t::none_k;

    dense_linux_aio_engine() noexcept = default;
    dense_linux_aio_engine(dense_linux_aio_engine const&) = delete;
    dense_linux_aio_engine& operator=(dense_linux_aio_engine const&) = delete;
    ~dense_linux_aio_engine() noexcept {
        if (aio_context_) syscall(SYS_io_destroy, aio_context_);
    }

    inline status_t open(int file_descriptor, std::size_t queue_depth, std::size_t buffer_stride,
                         allocator_t const& allocator) noexcept {
        file_descriptor_ = file_descriptor;
        status_t arena_status = arena_.reserve(queue_depth, buffer_stride, allocator);
        if (arena_status != status_t::success_k) return arena_status;
        control_blocks_ = control_blocks_tensor_t(control_blocks_allocator_t(allocator));
        control_pointers_ = control_pointers_tensor_t(control_pointers_allocator_t(allocator));
        events_ = events_tensor_t(events_allocator_t(allocator));
        if (!control_blocks_.reserve(queue_depth)) return status_t::bad_alloc_k;
        if (!control_pointers_.reserve(queue_depth)) return status_t::bad_alloc_k;
        if (!events_.reserve(queue_depth) || !events_.try_resize({queue_depth})) return status_t::bad_alloc_k;
        status_t completions_status = completions_.reserve(queue_depth, allocator);
        if (completions_status != status_t::success_k) return completions_status;
        aio_context_ = 0;
        long setup_result = syscall(SYS_io_setup, static_cast<unsigned>(queue_depth), &aio_context_);
        if (setup_result < 0) return status_t::device_error_k;
        return status_t::success_k;
    }

    inline std::byte* buffer(dense_staging_buffer_id_t index) noexcept { return arena_.buffer(index); }

    /** @brief Encodes the control block in place at push - matching the ring engines'
     *      encode-at-push discipline - so pump only submits; the staged fill is the
     *      control-block tensor's live shape. */
    inline bool push(dense_io_request_t request) noexcept {
        std::size_t staged = control_blocks_.extent(0);
        if (in_flight_ + staged == events_.extent(0)) return false;
        if (!control_blocks_.try_resize({staged + 1}) || !control_pointers_.try_resize({staged + 1})) return false;
        struct iocb& control = control_blocks_[staged];
        control = iocb {};
        control.aio_fildes = static_cast<u32_t>(file_descriptor_);
        control.aio_lio_opcode = IOCB_CMD_PREAD;
        control.aio_buf = reinterpret_cast<std::uintptr_t>(arena_.buffer(request.buffer));
        control.aio_nbytes = request.length;
        control.aio_offset = static_cast<std::int64_t>(request.offset);
        control.aio_data = request.tag;
        control_pointers_[staged] = &control;
        return true;
    }

    inline status_t pump(dense_io_wait_t wait) noexcept {
        std::size_t staged = control_blocks_.extent(0);
        if (staged) {
            long submitted = syscall(SYS_io_submit, aio_context_, static_cast<long>(staged), control_pointers_.data());
            if (submitted < 0) return status_t::device_error_k;
            in_flight_ += static_cast<std::size_t>(submitted);
            if (!control_blocks_.try_resize({0}) || !control_pointers_.try_resize({0})) return status_t::overflow_k;
        }
        if (!in_flight_) return status_t::success_k;
        // AIO surfaces no monitorable completion word, so there is no user-space fast
        // path: one io_getevents does it. A blocking reap returns ready events at once
        // and blocks only when none are in yet - a preceding min=0 probe is redundant.
        long reaped = reap_(wait == dense_io_wait_t::blocking_k ? 1 : 0);
        if (reaped < 0) return status_t::device_error_k;
        for (long i = 0; i != reaped; ++i)
            if (!completions_.complete(dense_io_completion_t {events_[i].data, events_[i].res}))
                return status_t::overflow_k;
        in_flight_ -= static_cast<std::size_t>(reaped);
        return status_t::success_k;
    }

    inline bool pop(dense_io_completion_t& completion) noexcept { return completions_.pop(completion); }

  private:
    inline long reap_(long minimum) noexcept {
        return syscall(SYS_io_getevents, aio_context_, minimum, static_cast<long>(events_.extent(0)), events_.data(),
                       nullptr);
    }

    int file_descriptor_ = -1;
    aio_context_t aio_context_ = 0;
    std::size_t in_flight_ = 0;
    control_blocks_tensor_t control_blocks_;
    control_pointers_tensor_t control_pointers_;
    events_tensor_t events_;
    dense_io_arena<allocator_t> arena_;
    dense_io_completions<allocator_t> completions_;
};

/**
 *  @brief Shared io_uring plumbing of the Linux ring engines: setup with graceful
 *      flag degradation, the ring mappings, submission-queue slot claims, resource
 *      registration, and completion-queue draining - parameterized by SQE & CQE
 *      strides at runtime, since NVMe passthrough doubles both.
 *
 *  Setup prefers @c IORING_SETUP_SINGLE_ISSUER | @c IORING_SETUP_DEFER_TASKRUN -
 *  the 6.1+ mode where completion work runs batched exactly when the owning thread
 *  asks - and falls back to a plain ring on older kernels. One consequence matters
 *  to callers: under @c DEFER_TASKRUN completions only materialize inside
 *  @c io_uring_enter, so spinning on the CQ tail from user space is futile; the
 *  @c posting() accessor tells the pump loop which discipline applies.
 */
/** @brief SQE geometry of one ring: 64-byte entries for block-layer reads, or the
 *      128-byte form whose inline command area carries NVMe passthrough. */
enum class dense_uring_form_t : u32_t {
    compact_k = 0,
    nvme_k = 1,
};

/** @brief Completion discipline of one ring: interrupt-driven CQEs, or device-queue
 *      polling via @c IORING_SETUP_IOPOLL - which demands direct I/O & poll queues. */
enum class dense_uring_reaping_t : u32_t {
    interrupted_k = 0,
    polled_k = 1,
};

/** @brief When CQEs become visible: @c immediate_k rings post them asynchronously -
 *      a waiter can monitor the CQ tail from user space - while @c deferred_k rings
 *      run completion work only inside @c io_uring_enter, the single-issuer
 *      deferred-task-run discipline, making tail spinning pointless. */
enum class dense_uring_posting_t : u32_t {
    immediate_k = 0,
    deferred_k = 1,
};

/** @brief Whether NVMe passthrough commands reference pre-registered arena buffers
 *      via @c IORING_URING_CMD_FIXED - honored by kernels 6.14 & newer, skipping the
 *      per-command page pinning - or pass plain addresses the kernel pins each time.
 *      Older kernels reject the flag per command, so this stays an explicit opt-in
 *      rather than a registration-success probe. */
enum class dense_uring_command_buffers_t : u32_t {
    plain_k = 0,
    fixed_k = 1,
};

/** @brief One @c mmap -ed ring region; unmaps itself, so teardown is member order. */
struct dense_uring_mapping_t {
    void* base = nullptr;
    std::size_t bytes = 0;

    dense_uring_mapping_t() noexcept = default;
    dense_uring_mapping_t(dense_uring_mapping_t const&) = delete;
    dense_uring_mapping_t& operator=(dense_uring_mapping_t const&) = delete;
    ~dense_uring_mapping_t() noexcept { reset(); }

    inline status_t map(int ring_descriptor, std::size_t map_bytes, u64_t region) noexcept {
        base = ::mmap(nullptr, map_bytes, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, ring_descriptor,
                      static_cast<off_t>(region));
        if (base == MAP_FAILED) return base = nullptr, status_t::device_error_k;
        bytes = map_bytes;
        return status_t::success_k;
    }
    inline void reset() noexcept {
        if (base) ::munmap(base, bytes);
        base = nullptr, bytes = 0;
    }
    inline u32_t* word(u32_t offset) const noexcept {
        return reinterpret_cast<u32_t*>(static_cast<std::byte*>(base) + offset);
    }
};

/** @brief Everything the submission side of one ring needs: the kernel-shared head &
 *      tail words, the index-order array, the wrap mask, and the SQE array. */
struct dense_uring_submissions_t {
    u32_t* head = nullptr;
    u32_t* tail = nullptr;
    u32_t* order = nullptr;
    u32_t mask = 0;
    std::byte* entries = nullptr;
    std::size_t stride = 0;
};

/** @brief The completion-side mirror: head & tail words, wrap mask, and CQE array. */
struct dense_uring_completions_t {
    u32_t* head = nullptr;
    u32_t* tail = nullptr;
    u32_t mask = 0;
    std::byte* entries = nullptr;
    std::size_t stride = 0;
};

struct dense_uring_ring_t {

    dense_uring_ring_t() noexcept = default;
    dense_uring_ring_t(dense_uring_ring_t const&) = delete;
    dense_uring_ring_t& operator=(dense_uring_ring_t const&) = delete;
    ~dense_uring_ring_t() noexcept {
        if (ring_descriptor_ >= 0) ::close(ring_descriptor_);
    }

    /** @brief Creates the ring & maps its regions.
     *
     *  @param[in] queue_depth Submission-queue entries; completions sized 2x by the kernel.
     *  @param[in] form SQE & CQE strides - compact, or doubled for NVMe passthrough.
     *  @param[in] reaping Interrupt-driven completions, or device-queue polling.
     */
    inline status_t setup(std::size_t queue_depth, dense_uring_form_t form, dense_uring_reaping_t reaping) noexcept {
        struct io_uring_params params;
        u32_t shape_flags = 0;
        if (form == dense_uring_form_t::nvme_k) shape_flags |= IORING_SETUP_SQE128 | IORING_SETUP_CQE32;
        if (reaping == dense_uring_reaping_t::polled_k) shape_flags |= IORING_SETUP_IOPOLL;
        u32_t preferred_flags = shape_flags | IORING_SETUP_SINGLE_ISSUER | IORING_SETUP_DEFER_TASKRUN;
        for (u32_t flags : {preferred_flags, shape_flags}) {
            params = io_uring_params {};
            params.flags = flags;
            long ring_file_descriptor = syscall(SYS_io_uring_setup, static_cast<unsigned>(queue_depth), &params);
            if (ring_file_descriptor >= 0) {
                ring_descriptor_ = static_cast<int>(ring_file_descriptor);
                break;
            }
            if (flags == shape_flags) return status_t::device_error_k;
        }
        reaping_ = reaping;
        posting_ = params.flags & IORING_SETUP_DEFER_TASKRUN ? dense_uring_posting_t::deferred_k //
                                                             : dense_uring_posting_t::immediate_k;
        submissions_.stride = form == dense_uring_form_t::nvme_k ? 128 : sizeof(struct io_uring_sqe);
        completions_.stride = form == dense_uring_form_t::nvme_k ? 32 : sizeof(struct io_uring_cqe);

        // Map the two rings & the SQE array. Modern kernels serve SQ & CQ from one mapping.
        std::size_t submission_bytes = params.sq_off.array + params.sq_entries * sizeof(u32_t);
        std::size_t completion_bytes = params.cq_off.cqes + params.cq_entries * completions_.stride;
        bool single_mmap = params.features & IORING_FEAT_SINGLE_MMAP;
        if (single_mmap) submission_bytes = max_of(submission_bytes, completion_bytes);
        status_t mapped = submission_mapping_.map(ring_descriptor_, submission_bytes, IORING_OFF_SQ_RING);
        if (mapped != status_t::success_k) return mapped;
        dense_uring_mapping_t const& completion_mapping = single_mmap ? submission_mapping_ : completion_mapping_;
        if (!single_mmap) {
            mapped = completion_mapping_.map(ring_descriptor_, completion_bytes, IORING_OFF_CQ_RING);
            if (mapped != status_t::success_k) return mapped;
        }
        mapped = entries_mapping_.map(ring_descriptor_, params.sq_entries * submissions_.stride, IORING_OFF_SQES);
        if (mapped != status_t::success_k) return mapped;

        submissions_.head = submission_mapping_.word(params.sq_off.head);
        submissions_.tail = submission_mapping_.word(params.sq_off.tail);
        submissions_.order = submission_mapping_.word(params.sq_off.array);
        submissions_.mask = *submission_mapping_.word(params.sq_off.ring_mask);
        submissions_.entries = static_cast<std::byte*>(entries_mapping_.base);
        completions_.head = completion_mapping.word(params.cq_off.head);
        completions_.tail = completion_mapping.word(params.cq_off.tail);
        completions_.mask = *completion_mapping.word(params.cq_off.ring_mask);
        completions_.entries = static_cast<std::byte*>(completion_mapping.base) + params.cq_off.cqes;
        pending_tail_ = *submissions_.tail;
        register_ring_descriptor_();
        return status_t::success_k;
    }

    /** @brief Pins the arena's buffers into the ring's registered-buffer table, one
     *      @c iovec per staging buffer, so @c READ_FIXED skips per-read page pinning. */
    inline status_t register_buffers(std::span<struct iovec const> vectors) noexcept {
        long registered = syscall(SYS_io_uring_register, ring_descriptor_, IORING_REGISTER_BUFFERS, vectors.data(),
                                  static_cast<unsigned>(vectors.size()));
        return registered < 0 ? status_t::device_error_k : status_t::success_k;
    }
    /** @brief Registers the one payload descriptor as fixed-file index 0, removing
     *      per-read descriptor refcounting. */
    inline status_t register_file(int file_descriptor) noexcept {
        std::int32_t descriptors[1] = {file_descriptor};
        long registered = syscall(SYS_io_uring_register, ring_descriptor_, IORING_REGISTER_FILES, descriptors, 1u);
        return registered < 0 ? status_t::device_error_k : status_t::success_k;
    }

    /** @brief Pins every buffer of @p arena into the ring's registered-buffer table -
     *      one @c iovec per staging buffer - building the transient iovec array through
     *      @p allocator, so even setup scratch honors the caller's memory domain. */
    template <typename allocator_type_>
    inline status_t register_arena(dense_io_arena<allocator_type_> const& arena, std::size_t buffer_count,
                                   allocator_type_ const& allocator) noexcept {
        using iovecs_allocator_t = typename std::allocator_traits<allocator_type_>::template rebind_alloc<struct iovec>;
        using iovecs_tensor_t = nk::tensor<struct iovec, iovecs_allocator_t, 1>;
        iovecs_tensor_t buffer_regions {iovecs_allocator_t(allocator)};
        if (!buffer_regions.reserve(buffer_count) || !buffer_regions.try_resize({buffer_count}))
            return status_t::bad_alloc_k;
        for (std::size_t i = 0; i != buffer_count; ++i)
            buffer_regions[i] = iovec {arena.buffer(static_cast<dense_staging_buffer_id_t>(i)), arena.stride};
        return register_buffers(std::span<struct iovec const> {buffer_regions.data(), buffer_count});
    }

    /** @brief Claims the next SQE slot, or returns null when the ring is full. The caller
     *      fills the returned entry in place; @ref enter publishes it. The claim cursor
     *      is @c pending_tail_ - not the kernel-visible tail - so a batch of claims
     *      lands in consecutive slots before one publication. */
    inline struct io_uring_sqe* next_submission() noexcept {
        u32_t head = std::atomic_ref<u32_t>(*submissions_.head).load(std::memory_order_acquire);
        if (pending_tail_ - head == submissions_.mask + 1) return nullptr;
        u32_t index = pending_tail_ & submissions_.mask;
        submissions_.order[index] = index;
        pending_tail_++;
        return reinterpret_cast<struct io_uring_sqe*>(submissions_.entries + index * submissions_.stride);
    }

    /** @brief Publishes claimed SQEs & performs one @c io_uring_enter with the given
     *      completion demand; @p min_complete of zero submits without blocking. */
    inline status_t enter(u32_t min_complete) noexcept {
        u32_t previous_tail = *submissions_.tail;
        u32_t to_submit = pending_tail_ - previous_tail;
        if (to_submit) std::atomic_ref<u32_t>(*submissions_.tail).store(pending_tail_, std::memory_order_release);
        // Polled rings only harvest device completions inside the syscall, and deferred
        // task-run rings only post them there - both must ask for events every time.
        bool getevents = min_complete || posting_ == dense_uring_posting_t::deferred_k || //
                         reaping_ == dense_uring_reaping_t::polled_k;
        if (!to_submit && !getevents) return status_t::success_k;
        u32_t enter_flags = getevents ? IORING_ENTER_GETEVENTS : 0u;
        int enter_descriptor = ring_descriptor_;
        if (ring_registered_) {
            enter_flags |= IORING_ENTER_REGISTERED_RING;
            enter_descriptor = static_cast<int>(registered_ring_index_);
        }
        long entered = syscall(SYS_io_uring_enter, enter_descriptor, to_submit, min_complete, enter_flags, nullptr,
                               0ull);
        if (entered < 0 && errno != EINTR && errno != EBUSY && errno != EAGAIN) return status_t::device_error_k;
        return status_t::success_k;
    }

    /** @brief Drains one CQE into @p completion; NVMe passthrough status decoding is the
     *      caller's job - this layer only surfaces the primary result word. */
    inline bool pop(dense_io_completion_t& completion) noexcept {
        u32_t head = *completions_.head;
        u32_t tail = std::atomic_ref<u32_t>(*completions_.tail).load(std::memory_order_acquire);
        if (head == tail) return false;
        struct io_uring_cqe const* entry = reinterpret_cast<struct io_uring_cqe const*>(
            completions_.entries + (head & completions_.mask) * completions_.stride);
        completion.tag = entry->user_data;
        completion.result = entry->res;
        std::atomic_ref<u32_t>(*completions_.head).store(head + 1, std::memory_order_release);
        return true;
    }

    /** @brief The full submit-and-reap discipline shared by every ring engine: nonblocking
     *      submits & returns; blocking on an async-posting ring pays one capped @p wait_type_
     *      monitor on the CQ tail before the blocking enter - a completion landing in that
     *      window reaps for zero syscalls - while deferred & polled rings fuse the pending
     *      batch and the wait into a single enter. */
    template <typename wait_type_>
    inline status_t pump(dense_io_wait_t wait) noexcept {
        if (wait == dense_io_wait_t::nonblocking_k) return enter(0);
        if (reaps_asynchronously_()) {
            status_t submitted = enter(0);
            if (submitted != status_t::success_k) return submitted;
            u32_t seen = completion_tail_seen_();
            wait_type_ {}(completion_tail_word_(), seen, 0u, fu::wait_capped_t {});
            if (completion_tail_seen_() != seen) return status_t::success_k;
        }
        return enter(1);
    }

  private:
    /** @brief The CQ tail word - the address a ForkUnion waiter monitors while the
     *      kernel completes reads in the background. */
    inline u32_t const* completion_tail_word_() const noexcept { return completions_.tail; }
    inline u32_t completion_tail_seen_() const noexcept {
        return std::atomic_ref<u32_t>(*completions_.tail).load(std::memory_order_acquire);
    }
    /** @brief True only when CQEs land in shared memory between syscalls - immediate
     *      posting under interrupt-driven reaping. Deferred task-run and IOPOLL rings
     *      post CQEs strictly inside @c io_uring_enter, so a user-space tail monitor
     *      can never observe them and the submit & wait pair fuses into one enter. */
    inline bool reaps_asynchronously_() const noexcept {
        return posting_ == dense_uring_posting_t::immediate_k && reaping_ == dense_uring_reaping_t::interrupted_k;
    }

    /** @brief Registers the ring descriptor so every enter addresses it by index,
     *      skipping the per-syscall descriptor-table lookup. Silent no-op fallback on
     *      pre-5.18 kernels, where the registration opcode is simply rejected. */
    inline void register_ring_descriptor_() noexcept {
        struct io_uring_rsrc_update ring_update {};
        ring_update.data = static_cast<u64_t>(ring_descriptor_);
        ring_update.offset = static_cast<u32_t>(-1); // The kernel assigns the index.
        long ring_registered = syscall(SYS_io_uring_register, ring_descriptor_, IORING_REGISTER_RING_FDS, &ring_update,
                                       1u);
        if (ring_registered < 0) return;
        registered_ring_index_ = ring_update.offset;
        ring_registered_ = true;
    }

    int ring_descriptor_ = -1;
    dense_uring_reaping_t reaping_ = dense_uring_reaping_t::interrupted_k;
    dense_uring_posting_t posting_ = dense_uring_posting_t::immediate_k;
    dense_uring_mapping_t submission_mapping_;
    dense_uring_mapping_t completion_mapping_;
    dense_uring_mapping_t entries_mapping_;
    dense_uring_submissions_t submissions_;
    dense_uring_completions_t completions_;
    u32_t pending_tail_ = 0;
    u32_t registered_ring_index_ = 0;
    bool ring_registered_ = false;
};

/**
 *  @brief The baseline io_uring engine: @c IORING_OP_READ against a file or block
 *      device. Kernel floor 6.1 for the single-issuer deferred-task-run mode,
 *      degrading to a plain 5.x ring transparently. Pair with an @c O_DIRECT
 *      descriptor for the page-cache-free path the format's block alignment is
 *      designed around.
 *
 *  The pump discipline: submit everything claimed, then - on rings where completions
 *  post asynchronously - give the ForkUnion waiter one capped monitor on the CQ tail
 *  before paying the blocking syscall; a completion arriving within that ~microsecond
 *  window costs zero syscalls to reap, since the CQ is drained entirely in user space.
 *  Under @c DEFER_TASKRUN the spin is skipped - completions only post inside
 *  @c io_uring_enter by design.
 */
template <typename wait_type_ = fu::preferred_yield_t, typename allocator_type_ = std::allocator<std::byte>>
struct dense_linux_uring_read_engine {
    using wait_t = wait_type_;
    using allocator_t = allocator_type_;

    inline static constexpr dense_io_capability_t capabilities_k = dense_io_capability_t::none_k;

    dense_linux_uring_read_engine() noexcept = default;
    dense_linux_uring_read_engine(dense_linux_uring_read_engine const&) = delete;
    dense_linux_uring_read_engine& operator=(dense_linux_uring_read_engine const&) = delete;

    inline status_t open(int file_descriptor, std::size_t queue_depth, std::size_t buffer_stride,
                         allocator_t const& allocator) noexcept {
        status_t arena_status = arena_.reserve(queue_depth, buffer_stride, allocator);
        if (arena_status != status_t::success_k) return arena_status;
        status_t ring_status = ring_.setup(queue_depth, dense_uring_form_t::compact_k,
                                           dense_uring_reaping_t::interrupted_k);
        if (ring_status != status_t::success_k) return ring_status;
        // Fixed-file registration composes with any opcode: plain reads keep their
        // unregistered buffers but still shed the per-submission descriptor refcount.
        return ring_.register_file(file_descriptor);
    }

    inline std::byte* buffer(dense_staging_buffer_id_t index) noexcept { return arena_.buffer(index); }

    inline bool push(dense_io_request_t request) noexcept {
        struct io_uring_sqe* entry = ring_.next_submission();
        if (!entry) return false;
        *entry = io_uring_sqe {};
        entry->opcode = IORING_OP_READ;
        entry->flags = IOSQE_FIXED_FILE;
        entry->fd = 0; // Fixed-file index, not a descriptor: the one registered payload.
        entry->addr = reinterpret_cast<std::uintptr_t>(arena_.buffer(request.buffer));
        entry->len = request.length;
        entry->off = request.offset;
        entry->user_data = request.tag;
        return true;
    }

    inline status_t pump(dense_io_wait_t wait) noexcept { return ring_.pump<wait_t>(wait); }

    inline bool pop(dense_io_completion_t& completion) noexcept { return ring_.pop(completion); }

  private:
    dense_uring_ring_t ring_;
    dense_io_arena<allocator_t> arena_;
};

/**
 *  @brief The default Linux fast path: @c IORING_OP_READ_FIXED over buffers pinned
 *      once via @c IORING_REGISTER_BUFFERS and the payload descriptor registered as
 *      fixed-file 0 - removing per-read page pinning & descriptor refcounting,
 *      roughly an 11% throughput gain at NVMe rates. Kernel floor matches the read
 *      engine; registration itself works on any ring.
 */
template <typename wait_type_ = fu::preferred_yield_t, typename allocator_type_ = std::allocator<std::byte>>
struct dense_linux_uring_fixed_engine {
    using wait_t = wait_type_;
    using allocator_t = allocator_type_;

    inline static constexpr dense_io_capability_t capabilities_k = dense_io_capability_t::fixed_buffers_k;

    dense_linux_uring_fixed_engine() noexcept = default;
    dense_linux_uring_fixed_engine(dense_linux_uring_fixed_engine const&) = delete;
    dense_linux_uring_fixed_engine& operator=(dense_linux_uring_fixed_engine const&) = delete;

    inline status_t open(int file_descriptor, std::size_t queue_depth, std::size_t buffer_stride,
                         allocator_t const& allocator) noexcept {
        status_t arena_status = arena_.reserve(queue_depth, buffer_stride, allocator);
        if (arena_status != status_t::success_k) return arena_status;
        status_t ring_status = ring_.setup(queue_depth, dense_uring_form_t::compact_k,
                                           dense_uring_reaping_t::interrupted_k);
        if (ring_status != status_t::success_k) return ring_status;
        status_t buffers_status = ring_.register_arena(arena_, queue_depth, allocator);
        if (buffers_status != status_t::success_k) return buffers_status;
        return ring_.register_file(file_descriptor);
    }

    inline std::byte* buffer(dense_staging_buffer_id_t index) noexcept { return arena_.buffer(index); }

    inline bool push(dense_io_request_t request) noexcept {
        struct io_uring_sqe* entry = ring_.next_submission();
        if (!entry) return false;
        *entry = io_uring_sqe {};
        entry->opcode = IORING_OP_READ_FIXED;
        entry->flags = IOSQE_FIXED_FILE;
        entry->fd = 0; // Fixed-file index, not a descriptor: the one registered payload.
        entry->addr = reinterpret_cast<std::uintptr_t>(arena_.buffer(request.buffer));
        entry->len = request.length;
        entry->off = request.offset;
        entry->buf_index = static_cast<std::uint16_t>(request.buffer);
        entry->user_data = request.tag;
        return true;
    }

    inline status_t pump(dense_io_wait_t wait) noexcept { return ring_.pump<wait_t>(wait); }

    inline bool pop(dense_io_completion_t& completion) noexcept { return ring_.pop(completion); }

  private:
    dense_uring_ring_t ring_;
    dense_io_arena<allocator_t> arena_;
};

/**
 *  @brief The polled variant: the fixed engine plus @c IORING_SETUP_IOPOLL -
 *      completions are harvested from the device's queues inside @c io_uring_enter
 *      instead of arriving by interrupt, worth ~21% on top of the fixed path.
 *      Requires an @c O_DIRECT descriptor on a device with poll-capable queues:
 *      @c nvme.poll_queues=N for real drives, @c poll_queues in the null_blk
 *      configfs recipe above for emulation; cross-check any setup with
 *      @c fio --ioengine=io_uring --hipri.
 *
 *  The pump is a bounded poll loop: non-blocking reap attempts, each empty
 *  iteration paced by one capped ForkUnion monitor - @c TPAUSE / @c WFET class
 *  waiters make this the energy-efficient busy-wait - with a final blocking
 *  enter as the backstop.
 */
template <typename wait_type_ = fu::preferred_yield_t, typename allocator_type_ = std::allocator<std::byte>>
struct dense_linux_uring_polled_engine {
    using wait_t = wait_type_;
    using allocator_t = allocator_type_;

    inline static constexpr dense_io_capability_t capabilities_k = dense_io_capability_t::fixed_buffers_k |
                                                                   dense_io_capability_t::polled_k;

    dense_linux_uring_polled_engine() noexcept = default;
    dense_linux_uring_polled_engine(dense_linux_uring_polled_engine const&) = delete;
    dense_linux_uring_polled_engine& operator=(dense_linux_uring_polled_engine const&) = delete;

    inline status_t open(int file_descriptor, std::size_t queue_depth, std::size_t buffer_stride,
                         allocator_t const& allocator) noexcept {
        status_t arena_status = arena_.reserve(queue_depth, buffer_stride, allocator);
        if (arena_status != status_t::success_k) return arena_status;
        status_t ring_status = ring_.setup(queue_depth, dense_uring_form_t::compact_k, dense_uring_reaping_t::polled_k);
        if (ring_status != status_t::success_k) return ring_status;
        status_t buffers_status = ring_.register_arena(arena_, queue_depth, allocator);
        if (buffers_status != status_t::success_k) return buffers_status;
        return ring_.register_file(file_descriptor);
    }

    inline std::byte* buffer(dense_staging_buffer_id_t index) noexcept { return arena_.buffer(index); }

    inline bool push(dense_io_request_t request) noexcept {
        struct io_uring_sqe* entry = ring_.next_submission();
        if (!entry) return false;
        *entry = io_uring_sqe {};
        entry->opcode = IORING_OP_READ_FIXED;
        entry->flags = IOSQE_FIXED_FILE;
        entry->fd = 0;
        entry->addr = reinterpret_cast<std::uintptr_t>(arena_.buffer(request.buffer));
        entry->len = request.length;
        entry->off = request.offset;
        entry->buf_index = static_cast<std::uint16_t>(request.buffer);
        entry->user_data = request.tag;
        return true;
    }

    inline status_t pump(dense_io_wait_t wait) noexcept { return ring_.pump<wait_t>(wait); }

    inline bool pop(dense_io_completion_t& completion) noexcept { return ring_.pop(completion); }

  private:
    dense_uring_ring_t ring_;
    dense_io_arena<allocator_t> arena_;
};

/**
 *  @brief NVMe passthrough engine: raw Read commands to the namespace character device
 *      @c /dev/ngXnY through 128-byte SQEs, bypassing the filesystem and the block
 *      layer - worth ~20-30% IOPS per core over the block path. Kernel floor 6.2 for
 *      unprivileged reads gated on the device node's file mode; a udev rule granting
 *      the service user read access to @c /dev/ng* suffices, no @c CAP_SYS_ADMIN.
 *
 *  The payload must live on the namespace directly - offsets translate to LBAs by a
 *  plain shift, so call @c configure with the namespace's LBA size before @c open.
 *  Discover & reshape the geometry with @c nvme-cli:
 *
 *  @code{.sh}
 *  nvme id-ns /dev/nvme0n1            # `lbads:9` = 512e, `lbads:12` = 4Kn
 *  nvme format /dev/nvme0n1 --lbaf=0  # reformat, DESTROYS the namespace
 *  @endcode
 *
 *  Deployment notes: on trusted storage servers boot with @c iommu=pt so DMA skips
 *  per-read IOMMU mapping - a measured ~4x IOPS cliff otherwise on mapping-bound
 *  paths. Completion results surface NVMe status: zero is success, positives are
 *  NVMe status codes, negatives remain kernel errors - @c pop folds both failure
 *  families into negative results for the state machine.
 */
template <typename wait_type_ = fu::preferred_yield_t, typename allocator_type_ = std::allocator<std::byte>>
struct dense_linux_uring_nvme_engine {
    using wait_t = wait_type_;
    using allocator_t = allocator_type_;

    inline static constexpr dense_io_capability_t capabilities_k = dense_io_capability_t::lba_addressing_k;
    inline static constexpr std::uint8_t nvme_read_opcode_k = 0x02;

    dense_linux_uring_nvme_engine() noexcept = default;
    dense_linux_uring_nvme_engine(dense_linux_uring_nvme_engine const&) = delete;
    dense_linux_uring_nvme_engine& operator=(dense_linux_uring_nvme_engine const&) = delete;

    /** @brief Presets the namespace LBA geometry - log2 of the LBA size, 9 for 512e,
     *      12 for 4Kn - which @c open cannot derive without an admin Identify pass, and
     *      the command-buffer mode, an explicit 6.14+ opt-in. */
    inline void configure(u32_t lba_shift, dense_uring_command_buffers_t command_buffers =
                                               dense_uring_command_buffers_t::plain_k) noexcept {
        lba_shift_ = lba_shift;
        command_buffers_ = command_buffers;
    }

    inline status_t open(int file_descriptor, std::size_t queue_depth, std::size_t buffer_stride,
                         allocator_t const& allocator) noexcept {
        file_descriptor_ = file_descriptor;
        status_t arena_status = arena_.reserve(queue_depth, buffer_stride, allocator);
        if (arena_status != status_t::success_k) return arena_status;
        int namespace_id = ioctl(file_descriptor, NVME_IOCTL_ID);
        if (namespace_id < 0) return status_t::device_error_k;
        namespace_id_ = static_cast<u32_t>(namespace_id);
        status_t ring_status = ring_.setup(queue_depth, dense_uring_form_t::nvme_k,
                                           dense_uring_reaping_t::interrupted_k);
        if (ring_status != status_t::success_k) return ring_status;
        if (command_buffers_ == dense_uring_command_buffers_t::fixed_k) {
            status_t buffers_status = ring_.register_arena(arena_, queue_depth, allocator);
            if (buffers_status != status_t::success_k) return buffers_status;
        }
        return status_t::success_k;
    }

    inline std::byte* buffer(dense_staging_buffer_id_t index) noexcept { return arena_.buffer(index); }

    inline bool push(dense_io_request_t request) noexcept {
        struct io_uring_sqe* entry = ring_.next_submission();
        if (!entry) return false;
        u64_t* entry_words = reinterpret_cast<u64_t*>(entry);
        for (std::size_t word = 0; word != 16; ++word) entry_words[word] = 0;
        entry->opcode = IORING_OP_URING_CMD;
        entry->fd = file_descriptor_;
        entry->cmd_op = NVME_URING_CMD_IO;
        entry->user_data = request.tag;
        if (command_buffers_ == dense_uring_command_buffers_t::fixed_k) {
            entry->uring_cmd_flags = IORING_URING_CMD_FIXED;
            entry->buf_index = static_cast<std::uint16_t>(request.buffer);
        }
        struct nvme_uring_cmd* command = reinterpret_cast<struct nvme_uring_cmd*>(entry->cmd);
        command->opcode = nvme_read_opcode_k;
        command->nsid = namespace_id_;
        command->addr = reinterpret_cast<std::uintptr_t>(arena_.buffer(request.buffer));
        command->data_len = request.length;
        u64_t first_block = request.offset >> lba_shift_;
        u32_t block_count = request.length >> lba_shift_;
        command->cdw10 = static_cast<u32_t>(first_block);
        command->cdw11 = static_cast<u32_t>(first_block >> 32);
        command->cdw12 = block_count - 1;
        return true;
    }

    inline status_t pump(dense_io_wait_t wait) noexcept { return ring_.pump<wait_t>(wait); }

    /** @brief Rewrites the NVMe convention - zero result means the full transfer - into
     *      the byte-count convention the state machine consumes. */
    inline bool pop(dense_io_completion_t& completion) noexcept {
        if (!ring_.pop(completion)) return false;
        if (completion.result > 0) completion.result = -EIO; // NVMe status codes are failures.
        return true;
    }

  private:
    int file_descriptor_ = -1;
    u32_t namespace_id_ = 0;
    u32_t lba_shift_ = 9;
    dense_uring_command_buffers_t command_buffers_ = dense_uring_command_buffers_t::plain_k;
    dense_uring_ring_t ring_;
    dense_io_arena<allocator_t> arena_;
};

#endif // defined(__linux__)

#if defined(_WIN32)

/**
 *  @brief Windows I/O Rings engine - the closest io_uring sibling: shared rings,
 *      user-mode completion reaping, batched submission, and registered buffers &
 *      files mirroring the Linux fixed engine. Requires Windows 11 22H2 for
 *      @c IORING_VERSION_3; pair with @c FILE_FLAG_NO_BUFFERING handles, whose
 *      alignment contract matches this format: offsets, lengths & buffers aligned
 *      to the volume sector size - 512 or 4096, query via @c FILE_STORAGE_INFO.
 *
 *  Registration ops are themselves asynchronous SQEs with implicit barrier
 *  semantics - later entries wait for them - so @c open enqueues both and lets the
 *  first real read carry the dependency. Note WSL2 cannot substitute for this
 *  engine's testing: its storage arrives via a paravirtual SCSI path, so neither
 *  IoRing nor NVMe character devices exist there; only native Windows exercises
 *  this code.
 */
template <typename wait_type_ = fu::preferred_yield_t, typename allocator_type_ = std::allocator<std::byte>>
struct dense_windows_ioring_engine {
    using wait_t = wait_type_;
    using allocator_t = allocator_type_;

    inline static constexpr dense_io_capability_t capabilities_k = dense_io_capability_t::fixed_buffers_k;

    dense_windows_ioring_engine() noexcept = default;
    dense_windows_ioring_engine(dense_windows_ioring_engine const&) = delete;
    dense_windows_ioring_engine& operator=(dense_windows_ioring_engine const&) = delete;
    ~dense_windows_ioring_engine() noexcept {
        if (ring_) CloseIoRing(ring_);
    }

    inline status_t open(HANDLE file_handle, std::size_t queue_depth, std::size_t buffer_stride,
                         allocator_t const& allocator) noexcept {
        status_t arena_status = arena_.reserve(queue_depth, buffer_stride, allocator);
        if (arena_status != status_t::success_k) return arena_status;
        IORING_CREATE_FLAGS flags {IORING_CREATE_REQUIRED_FLAGS_NONE, IORING_CREATE_ADVISORY_FLAGS_NONE};
        HRESULT created = CreateIoRing(IORING_VERSION_3, flags, static_cast<UINT32>(queue_depth),
                                       static_cast<UINT32>(queue_depth * 2), &ring_);
        if (!SUCCEEDED(created)) return status_t::device_error_k;
        buffer_infos_ = buffer_infos_tensor_t(buffer_infos_allocator_t(allocator));
        if (!buffer_infos_.reserve(queue_depth) || !buffer_infos_.try_resize({queue_depth}))
            return status_t::bad_alloc_k;
        for (std::size_t i = 0; i != queue_depth; ++i)
            buffer_infos_[i] = IORING_BUFFER_INFO {arena_.buffer(static_cast<dense_staging_buffer_id_t>(i)),
                                                   static_cast<UINT32>(arena_.stride)};
        HRESULT buffers = BuildIoRingRegisterBuffers(ring_, static_cast<UINT32>(queue_depth), buffer_infos_.data(), 0);
        if (!SUCCEEDED(buffers)) return status_t::device_error_k;
        file_handle_ = file_handle;
        HRESULT files = BuildIoRingRegisterFileHandles(ring_, 1, &file_handle_, 0);
        return SUCCEEDED(files) ? status_t::success_k : status_t::device_error_k;
    }

    inline std::byte* buffer(dense_staging_buffer_id_t index) noexcept { return arena_.buffer(index); }

    inline bool push(dense_io_request_t request) noexcept {
        IORING_BUFFER_REF buffer_ref = IoRingBufferRefFromIndexAndOffset(request.buffer, 0);
        IORING_HANDLE_REF handle_ref = IoRingHandleRefFromIndex(0);
        HRESULT built = BuildIoRingReadFile(ring_, handle_ref, buffer_ref, request.length, request.offset, request.tag,
                                            IOSQE_FLAGS_NONE);
        return SUCCEEDED(built);
    }

    inline status_t pump(dense_io_wait_t wait) noexcept {
        UINT32 submitted = 0;
        bool blocking = wait == dense_io_wait_t::blocking_k;
        HRESULT result = SubmitIoRing(ring_, blocking ? 1u : 0u, blocking ? INFINITE : 0u, &submitted);
        return SUCCEEDED(result) ? status_t::success_k : status_t::device_error_k;
    }

    inline bool pop(dense_io_completion_t& completion) noexcept {
        IORING_CQE entry;
        if (PopIoRingCompletion(ring_, &entry) != S_OK) return false;
        completion.tag = entry.UserData;
        completion.result = SUCCEEDED(entry.ResultCode) ? static_cast<std::int64_t>(entry.Information)
                                                        : -static_cast<std::int64_t>(1);
        return true;
    }

  private:
    using buffer_infos_allocator_t =
        typename std::allocator_traits<allocator_type_>::template rebind_alloc<IORING_BUFFER_INFO>;
    using buffer_infos_tensor_t = nk::tensor<IORING_BUFFER_INFO, buffer_infos_allocator_t, 1>;

    HANDLE file_handle_ = nullptr;
    HIORING ring_ = nullptr;
    dense_io_arena<allocator_t> arena_;
    buffer_infos_tensor_t buffer_infos_;
};

#endif // defined(_WIN32)

#pragma endregion Engines

#pragma region State Machine

/** @brief Progress of one in-flight traversal through its slot's lifecycle. */
enum class dense_nvme_stage_t : std::uint8_t {
    /** @brief The slot is free; @c reset arms it with the next pending query. */
    idle_k,
    /** @brief Fetches are outstanding; completions drive the traversal forward. */
    traversing_k,
    /** @brief The traversal converged; results await copying out before recycling. */
    finished_k,
};

/**
 *  @brief One in-flight query's entire mutable state: the scored frontier, the running
 *      results, the exact visited set, and the in-flight read counter. A plain struct
 *      of caller-allocated containers - the state machine in the index advances it,
 *      the engine never sees it.
 */
template <                                               //
    nk_dtype_t vector_dtype_ = nk_bf16_k,                //
    typename allocator_type_ = std::allocator<std::byte> //
    >
struct dense_nvme_slot {

    using scalar_t = typename nk::type_for<vector_dtype_>::type;
    using vector_view_t = nk::vector_view<scalar_t>;
    using vector_norms_t = usearch::vector_norms<typename scalar_t::dot_result_t>;

    /** @brief Default integer type to use for loops and indexing at sub-index scale. */
    using small_size_t = std::uint32_t;

    using allocator_t = allocator_type_;
    using candidates_allocator_t =
        typename std::allocator_traits<allocator_t>::template rebind_alloc<dense_scored_node_t>;
    /** @brief The hash-set allocates raw bytes and places ids itself, so it demands a byte allocator. */
    using visits_allocator_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<std::byte>;

    using results_array_t = sorted_buffer<dense_scored_node_t, dense_closer_scored_node_t, candidates_allocator_t>;
    /** @brief 8-ary: eight 8-byte scored nodes fill one cache line per level scan. */
    using frontier_queue_t = max_heap_dary<8, dense_scored_node_t, dense_further_scored_node_t, candidates_allocator_t>;
    using visited_set_t = growing_hash_set<dense_node_id_t, std::hash<dense_node_id_t>, visits_allocator_t>;

    dense_nvme_stage_t stage = dense_nvme_stage_t::idle_k;
    /** @brief Reads issued and not yet completed for this slot; the slot only advances
     *      to its next hop when this drains to zero. */
    small_size_t inflight_reads = 0;

    /** @brief The caller's query row; must outlive the batch. */
    vector_view_t query;
    vector_norms_t query_norms {};
    /** @brief Which batch row this slot is serving - indexes the output arrays. */
    small_size_t query_index = 0;
    /** @brief This search's result-pool breadth - at most the reserved capacity, which
     *      container growth rounding can silently exceed; the traversal trusts only this. */
    small_size_t pool_limit = 0;

    /** @brief Nodes we've already scored, kept ascending as the best results so far. */
    results_array_t results;
    /** @brief Nodes we consider promising and want to expand next. */
    frontier_queue_t frontier;
    /** @brief Nodes already fetched for scoring; exact by design - the recall ceiling
     *      of a lossy filter is not worth its memory. */
    visited_set_t visited;
    /** @brief Leased buffer indexes of score completions awaiting a quad flush - drained
     *      every fourth arrival, and once more when the hop's in-flight reads reach zero,
     *      so deferral adds no latency. Stashing the index alone keeps the record bytes
     *      where DMA left them until the one moment the quad kernel reads them. */
    fu::limited_array<dense_staging_buffer_id_t, 4> pending_scores;
    /** @brief Pseudo-random generator for picking this slot's traversal starting points. */
    splitmix_u32_t random_generator;

    dense_nvme_slot() noexcept = default;
    dense_nvme_slot(dense_nvme_slot&&) noexcept = default;
    dense_nvme_slot(dense_nvme_slot const&) = delete;
    dense_nvme_slot& operator=(dense_nvme_slot const&) = delete;

    /** @brief Re-seats every container on @p allocator & sizes it for @p expansion results
     *      and the visited volume one traversal can touch. Slots default-construct so
     *      arrays of them can be sized first; this late binding is what places their
     *      queues in the caller's memory domain. Called once per context, never per query. */
    inline status_t reserve(std::size_t expansion, std::size_t connectivity_ceiling,
                            candidates_allocator_t const& allocator) noexcept {
        results = results_array_t(allocator);
        frontier = frontier_queue_t(allocator);
        visited = visited_set_t(visits_allocator_t(allocator));
        if (!results.reserve(expansion)) return status_t::bad_alloc_k;
        if (!frontier.reserve(expansion * connectivity_ceiling)) return status_t::bad_alloc_k;
        if (!visited.reserve(expansion * connectivity_ceiling * 2)) return status_t::bad_alloc_k;
        return status_t::success_k;
    }

    /** @brief Recycles the slot for @p new_query without touching any capacity. */
    inline void reset(vector_view_t new_query, vector_norms_t new_query_norms, small_size_t new_query_index,
                      small_size_t new_pool_limit, u64_t seed) noexcept {
        stage = dense_nvme_stage_t::traversing_k;
        inflight_reads = 0;
        query = new_query;
        query_norms = new_query_norms;
        query_index = new_query_index;
        pool_limit = new_pool_limit;
        results.clear();
        frontier.clear();
        visited.clear();
        pending_scores.clear();
        random_generator = splitmix_u32_t(seed);
    }
};

/**
 *  @brief Per-thread traversal harness: the engine with its staging arena, the free
 *      buffer-index stack, and the slot array - everything one thread needs to keep
 *      many traversals in flight. Constructed once, recycled across batches; owns
 *      no index state whatsoever.
 */
template <                                                                  //
    nk_dtype_t vector_dtype_ = nk_bf16_k,                                   //
    typename engine_type_ = dense_posix_sync_engine<fu::preferred_yield_t>, //
    typename allocator_type_ = std::allocator<std::byte>                    //
    >
struct dense_nvme_context {

    using engine_t = engine_type_;
    using allocator_t = allocator_type_;
    using slot_t = dense_nvme_slot<vector_dtype_, allocator_t>;
    using buffer_ids_allocator_t =
        typename std::allocator_traits<allocator_t>::template rebind_alloc<dense_staging_buffer_id_t>;
    using slots_allocator_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<slot_t>;
    using buffer_ids_tensor_t = nk::tensor<dense_staging_buffer_id_t, buffer_ids_allocator_t, 1>;
    using slots_array_t = fu::dynamic_array<slot_t, slots_allocator_t>;

    static_assert(dense_io_engine<engine_t>);

    engine_t engine;
    slots_array_t slots;

    /** @brief Stack of free arena buffer indexes; the live shape is the stack depth,
     *      so no separate top exists. The memory itself belongs to the engine - the
     *      state machine only leases indexes. */
    buffer_ids_tensor_t free_buffers;

    /** @brief The reserved result-pool ceiling; per-search pools are capped by it. */
    std::size_t expansion_capacity = 0;

    /** @brief Total one-to-one distance computations across the context's lifetime. */
    std::size_t computed_distances = 0;
    /** @brief Frontier candidates whose neighbor lists were fetched & expanded. */
    std::size_t expanded_members = 0;
    /** @brief Reads issued across the context's lifetime, refetches included. */
    std::size_t issued_reads = 0;

    dense_nvme_context(allocator_t const& allocator = {}) noexcept
        : engine(), slots(slots_allocator_t(allocator)), free_buffers(buffer_ids_allocator_t(allocator)),
          allocator_(allocator) {}
    dense_nvme_context(dense_nvme_context const&) = delete;
    dense_nvme_context& operator=(dense_nvme_context const&) = delete;

    /** @brief The allocator every slot & the engine inherit - exposed so the index can
     *      thread it into @c engine.open, keeping NUMA-domain placement intact. */
    inline allocator_t const& allocator() const noexcept { return allocator_; }

    /** @brief Claims every slot & queue this context will ever use; the engine's arena
     *      is claimed separately by @c engine.open with the same allocator.
     *
     *  @param[in] slot_count Traversals kept in flight simultaneously.
     *  @param[in] expansion Result-pool size each traversal converges on.
     *  @param[in] connectivity_ceiling Most neighbor ids any record may carry.
     */
    inline status_t reserve(std::size_t slot_count, std::size_t expansion, std::size_t connectivity_ceiling) noexcept {
        // A slot's transient peak holds its expansion record while emitting up to
        // connectivity scoring fetches, so each slot budgets connectivity + 1 buffers.
        std::size_t depth = slot_count * (max_of<std::size_t>(connectivity_ceiling, 1) + 1);
        if (!free_buffers.reserve(depth) || !free_buffers.try_resize({depth})) return status_t::bad_alloc_k;
        for (std::size_t i = 0; i != depth; ++i)
            free_buffers[i] = static_cast<dense_staging_buffer_id_t>(depth - 1 - i);

        if (!slots.try_resize(slot_count)) return status_t::bad_alloc_k;
        for (std::size_t i = 0; i != slot_count; ++i) {
            status_t slot_status = slots[i].reserve(expansion, connectivity_ceiling,
                                                    typename slot_t::candidates_allocator_t(allocator_));
            if (slot_status != status_t::success_k) return slot_status;
        }
        expansion_capacity = expansion;
        return status_t::success_k;
    }

    inline dense_staging_buffer_id_t acquire_buffer() noexcept {
        std::size_t depth = free_buffers.extent(0);
        dense_staging_buffer_id_t leased = free_buffers[depth - 1];
        return (void)free_buffers.try_resize({depth - 1}), leased;
    }
    inline void release_buffer(dense_staging_buffer_id_t index) noexcept {
        std::size_t depth = free_buffers.extent(0);
        (void)free_buffers.try_resize({depth + 1}); // Never exceeds the reservation it drained from.
        free_buffers[depth] = index;
    }

  private:
    [[no_unique_address]] allocator_t allocator_;
};

#pragma endregion State Machine

#pragma region Index

/**
 *  @brief Vector Search index for pre-constructed read-only graphs on block storage.
 *
 *  The class itself is a thin frozen directory: the geometry, the per-node byte offsets,
 *  and the descriptors of the payload. All traversal state lives in caller-owned
 *  contexts, and all I/O flows through the context's engine - so the same immutable
 *  index instance serves any number of threads, each pumping its own context.
 *
 *  @section dense_nvme_index_traversal Traversal Shape
 *
 *  Every fetched record carries both the vector and the neighbor list, so scoring a
 *  node costs one read. Expanding a node re-fetches its record to recover the neighbor
 *  list - one extra read per expanded node, roughly a 1/connectivity overhead - keeping
 *  frontier entries at eight bytes instead of dragging whole neighbor lists through the
 *  queues. Within one traversal, hops are sequential by nature; across the slots of a
 *  context, everything overlaps.
 *
 *  @section dense_nvme_index_passthru Passthrough Caveat
 *
 *  The LBA-addressed engine cannot serve @c open's metadata loading - character
 *  devices reject @c pread - so a passthrough deployment loads the directory from a
 *  sibling file or block-device view of the same namespace. This class currently
 *  targets the file & block-device engines end-to-end.
 */
template <                                                                  //
    nk_dtype_t vector_dtype_ = nk_bf16_k,                                   //
    nk_kernel_kind_t kernel_kind_ = nk_kernel_angular_k,                    //
    nk_capability_t capability_ = nk_cap_icelake_k,                         //
    typename engine_type_ = dense_posix_sync_engine<fu::preferred_yield_t>, //
    typename offsets_allocator_type_ = std::allocator<std::byte>            //
    >
struct dense_nvme_index {

    inline static constexpr nk_dtype_t vector_dtype_k = vector_dtype_;
    inline static constexpr nk_kernel_kind_t kernel_kind_k = kernel_kind_;

    using scalar_t = typename nk::type_for<vector_dtype_>::type;
    using vector_product_result_t = typename scalar_t::dot_result_t;
    using distance_t = f32_t;

    /** @brief Default integer type to use for loops and indexing at sub-index scale. */
    using small_size_t = std::uint32_t;
    /** @brief Node identifier within this sub-index - the class-local spelling of @c dense_node_id_t. */
    using node_id_t = dense_node_id_t;

    using vector_view_t = nk::vector_view<scalar_t>;
    using vectors_view_t = nk::matrix_view<scalar_t>;
    using vector_norms_t = vector_norms<vector_product_result_t>;

    using engine_t = engine_type_;
    using context_t = dense_nvme_context<vector_dtype_, engine_t, offsets_allocator_type_>;
    using slot_t = typename context_t::slot_t;

    using offsets_allocator_t = typename std::allocator_traits<offsets_allocator_type_>::template rebind_alloc<u64_t>;
    using offsets_tensor_t = nk::tensor<u64_t, offsets_allocator_t, 1>;

    /** @brief The capability bit of the distance kit the chain actually picked - lower than
     *      @c capability_ when no SIMD kit is adapted for this dtype & metric pair yet. */
    inline static constexpr nk_capability_t picked_quads_capability_k =
        dense_quads_chain<kernel_kind_, scalar_t, capability_>::picked_k;
    /** @brief The strongest @c synchronization_t contract this design honors: the payload
     *      is immutable, so arbitrary readers share one instance with no fences at all. */
    inline static constexpr synchronization_t synchronization_ceiling_k = synchronization_t::immutable_k;

    dense_nvme_index(offsets_allocator_type_ const& allocator = {}) noexcept
        : offsets_(offsets_allocator_t(allocator)) {}
    dense_nvme_index(dense_nvme_index const&) = delete;
    dense_nvme_index& operator=(dense_nvme_index const&) = delete;
    ~dense_nvme_index() noexcept { close(); }

    /** @brief Loads the directory - header & offsets - through ordinary buffered reads,
     *      and keeps a separate descriptor for the engines, optionally @c O_DIRECT.
     *
     *  @param[in] path Payload file or block device holding the serialized graph.
     *  @param[in] caching Routes the engine-side descriptor through the page cache,
     *      or around it - @c O_DIRECT on Linux, @c F_NOCACHE on macOS.
     */
    inline status_t open(char const* path, dense_io_caching_t caching = dense_io_caching_t::cached_k) noexcept {
#if defined(_WIN32)
        (void)path, (void)caching;
        return status_t::uninitialized_k; // Windows file plumbing arrives with its engine's bring-up.
#else
        dense_posix_descriptor_t metadata {::open(path, O_RDONLY)};
        if (metadata.file_descriptor < 0) return status_t::device_error_k;
        dense_index_header_t header;
        ssize_t header_bytes = ::pread(metadata.file_descriptor, &header, sizeof(header), 0);
        if (header_bytes != sizeof(header) || header.magic != dense_index_header_t::magic_k ||
            header.version != dense_index_header_t::version_k)
            return status_t::malformed_k;
        if (header.dtype != static_cast<u32_t>(vector_dtype_) || header.kernel != static_cast<u32_t>(kernel_kind_))
            return status_t::config_mismatch_k;

        dimensions_ = header.dimensions;
        connectivity_ceiling_ = header.connectivity;
        std::size_t offsets_count = std::size_t(header.node_count) + 1;
        if (!offsets_.reserve(offsets_count) || !offsets_.try_resize({offsets_count})) return status_t::bad_alloc_k;
        std::size_t offsets_bytes = dense_nvme_layout_t::offsets_bytes(header.node_count);
        ssize_t read_bytes = ::pread(metadata.file_descriptor, offsets_.data(), offsets_bytes,
                                     dense_nvme_layout_t::offsets_begin_k);
        if (read_bytes != static_cast<ssize_t>(offsets_bytes)) return status_t::malformed_k;

        record_capacity_ = 0;
        for (small_size_t i = 0; i != header.node_count; ++i) {
            u64_t begin = offsets_[i], end = offsets_[i + 1];
            if (end < begin || begin % dense_nvme_block_k) return status_t::malformed_k;
            record_capacity_ = max_of<std::size_t>(record_capacity_, end - begin);
        }
        int flags = O_RDONLY;
#if defined(__linux__)
        if (caching == dense_io_caching_t::direct_k) flags |= O_DIRECT;
#endif
        data_descriptor_ = ::open(path, flags);
        if (data_descriptor_ < 0) return status_t::device_error_k;
#if defined(__APPLE__)
        if (caching == dense_io_caching_t::direct_k) fcntl(data_descriptor_, F_NOCACHE, 1);
#endif
        return status_t::success_k;
#endif
    }

    inline void close() noexcept {
#if !defined(_WIN32)
        if (data_descriptor_ >= 0) ::close(data_descriptor_), data_descriptor_ = -1;
#endif
    }

    /** @brief Published nodes - the offsets directory minus its trailing sentinel. */
    inline small_size_t size() const noexcept {
        std::size_t offsets_count = offsets_.extent(0);
        return offsets_count ? static_cast<small_size_t>(offsets_count - 1) : 0;
    }
    inline small_size_t dimensions() const noexcept { return dimensions_; }
    inline small_size_t connectivity_ceiling() const noexcept { return connectivity_ceiling_; }
    /** @brief Longest record in the payload - the staging-buffer stride engines need. */
    inline std::size_t record_capacity() const noexcept { return record_capacity_; }
    inline int data_descriptor() const noexcept { return data_descriptor_; }

    /** @brief Sizing of one traversal harness; passed to @c reserve. */
    struct context_config_t {
        /** @brief Traversals kept in flight simultaneously; must be non-zero. */
        std::size_t slot_count = 0;
        /** @brief Ceiling on the per-traversal result pool; zero derives twice the
         *      connectivity. Searches may ask for less, never for more. */
        small_size_t expansion_factor = 0;
    };

    /** @brief Per-search tuning knobs; zeros mean "derive from connectivity". */
    struct search_config_t {
        /** @brief Breadth of each traversal's result pool; zero derives twice the
         *      connectivity, and the context's reserved ceiling always caps it. */
        small_size_t expansion_factor = 0;
    };

    /** @brief Outcome & telemetry of one @c search() call; matches land in the caller's spans. */
    struct search_result_t {
        /** @brief @c success_k, or why the batch could not run at all. */
        status_t status = status_t::success_k;
        /** @brief Frontier candidates whose records were re-fetched & expanded. */
        std::size_t expanded_members = 0;
        /** @brief Total distance computations across the batch. */
        std::size_t computed_distances = 0;
        /** @brief Reads issued for this batch, expansion refetches included. */
        std::size_t issued_reads = 0;

        explicit operator bool() const noexcept { return status == status_t::success_k; }
    };

    /** @brief Sizes @p context per @p config against this payload, then opens its engine
     *      on the data descriptor - threading the context's allocator through, so a
     *      NUMA-domain allocator places the engine's arena alongside the slots it serves. */
    inline status_t reserve(context_t& context, context_config_t config) noexcept {
        if (!config.slot_count) return status_t::config_mismatch_k;
        std::size_t expansion = derived_expansion_(config.expansion_factor);
        status_t reserved = context.reserve(config.slot_count, expansion, connectivity_ceiling_);
        if (reserved != status_t::success_k) return reserved;
        std::size_t depth = config.slot_count * (max_of<std::size_t>(connectivity_ceiling_, 1) + 1);
        return context.engine.open(data_descriptor_, depth, record_capacity_, context.allocator());
    }

    /**
     *  @brief Finds the nearest neighbors of every row of @p queries, overlapping all
     *      traversals through the context's engine.
     *
     *  The payload is immutable, so any number of threads may search one index
     *  simultaneously - each through its own context. Within a call the slots pipeline
     *  freely: a fast query's slot is recycled for the next pending row while slower
     *  traversals are still in flight, so the engine's queue stays full until the
     *  batch drains. @ref dense_nvme_concurrency
     *
     *  @param[in] queries The query batch, shape @b [count,dimensions], in the index's
     *      own dtype & frozen dimensionality.
     *  @param[out] results Caller-owned matrix of scored neighbors, shape
     *      @b [count,wanted] - the trailing extent is the per-query request - each row
     *      ascending by distance, the nearest first.
     *  @param[out] result_counts Kept entries per row, one slot per query - fewer than
     *      the requested width only on graphs smaller than the request.
     *  @param[inout] context The calling thread's harness - slots, engine & telemetry -
     *      sized by a prior @c reserve.
     *  @param[in] config Optional tuning knobs; zeros mean "derive from connectivity".
     */
    search_result_t search(vectors_view_t queries, nk::matrix_span<dense_scored_node_t> results,
                           nk::vector_span<small_size_t> result_counts, context_t& context,
                           search_config_t config = {}) noexcept {

        search_result_t outcome;
        if (data_descriptor_ < 0) return outcome.status = status_t::uninitialized_k, outcome;
        small_size_t total_queries = static_cast<small_size_t>(queries.extent(0));
        small_size_t wanted = static_cast<small_size_t>(results.extent(1));
        if (results.extent(0) != total_queries || result_counts.size() != total_queries)
            return outcome.status = status_t::config_mismatch_k, outcome;
        if (!size()) {
            for (small_size_t i = 0; i != total_queries; ++i) result_counts[i] = 0;
            return outcome;
        }
        // The pool never underflows the request nor overflows the reserved ceiling.
        small_size_t pool_limit = static_cast<small_size_t>(min_of<std::size_t>(
            max_of<std::size_t>(derived_expansion_(config.expansion_factor), wanted), context.expansion_capacity));

        std::size_t expanded_before = context.expanded_members;
        std::size_t computed_before = context.computed_distances;
        std::size_t issued_before = context.issued_reads;
        small_size_t next_query = 0;
        small_size_t finished_queries = 0;

        while (finished_queries != total_queries) {
            status_t armed = arm_slots_(context, queries, pool_limit, next_query, total_queries);
            if (armed != status_t::success_k) return outcome.status = armed, outcome;

            status_t pumped = context.engine.pump(dense_io_wait_t::blocking_k);
            if (pumped != status_t::success_k) return outcome.status = pumped, outcome;

            dense_io_completion_t completion;
            while (context.engine.pop(completion)) {
                status_t consumed = consume_(context, completion);
                if (consumed != status_t::success_k) return outcome.status = consumed, outcome;
            }

            harvest_slots_(context, results, result_counts, wanted, finished_queries);
        }
        outcome.expanded_members = context.expanded_members - expanded_before;
        outcome.computed_distances = context.computed_distances - computed_before;
        outcome.issued_reads = context.issued_reads - issued_before;
        return outcome;
    }

  private:
    /** @brief The zeros-mean-derive rule shared by reserve- & search-side configs:
     *      twice the connectivity, the classic breadth-vs-degree balance. */
    inline std::size_t derived_expansion_(small_size_t requested) const noexcept {
        return requested ? requested : 2 * max_of<std::size_t>(connectivity_ceiling_, 1);
    }

    /** @brief Trampoline for @c usearch::dense_norms with hardened types. */
    inline static vector_norms_t dense_norms(vector_view_t vec) noexcept {
        return usearch::dense_norms<capability_, scalar_t>(vec);
    }
    /** @brief Trampoline for @c usearch::distance with hardened types. */
    inline static distance_t distance(                   //
        vector_view_t query, vector_norms_t query_norms, //
        vector_view_t candidate, vector_norms_t candidate_norms) noexcept {
        return usearch::distance<kernel_kind_, capability_, scalar_t>( //
            query, query_norms, candidate, candidate_norms);
    }
    /** @brief Trampoline for @c usearch::distances with hardened types. */
    inline static std::array<distance_t, 4> distances(   //
        vector_view_t query, vector_norms_t query_norms, //
        std::array<vector_view_t, 4> candidates, std::array<vector_norms_t, 4> candidates_norms) noexcept {
        return usearch::distances<kernel_kind_, capability_, scalar_t>( //
            query, query_norms, candidates, candidates_norms);
    }

    /** @brief Issues one record fetch on behalf of @p slot_index, leasing an arena index. */
    inline status_t emit_(context_t& context, dense_traversal_slot_id_t slot_index, node_id_t node,
                          dense_io_purpose_t purpose) noexcept {
        dense_staging_buffer_id_t buffer_index = context.acquire_buffer();
        dense_io_tag_t tag;
        tag.slot = slot_index;
        tag.buffer = buffer_index;
        tag.purpose = static_cast<u32_t>(purpose);
        u64_t begin = offsets_[node], end = offsets_[node + 1];
        dense_io_request_t request;
        request.offset = begin;
        request.length = static_cast<u32_t>(end - begin);
        request.buffer = buffer_index;
        request.tag = tag.packed();
        context.slots[slot_index].inflight_reads++;
        context.issued_reads++;
        // The budget - depth = slots x (connectivity + 1) - guarantees the push always lands.
        return context.engine.push(request) ? status_t::success_k : status_t::overflow_k;
    }

    /** @brief Arms every idle slot with the next pending query row & seeds its traversal -
     *      the admission half of the pump loop. */
    inline status_t arm_slots_(context_t& context, vectors_view_t queries, small_size_t pool_limit,
                               small_size_t& next_query, small_size_t total_queries) noexcept {
        for (dense_traversal_slot_id_t slot_index = 0;
             slot_index != context.slots.size() && next_query != total_queries; ++slot_index) {
            slot_t& slot = context.slots[slot_index];
            if (slot.stage != dense_nvme_stage_t::idle_k) continue;
            vector_view_t query = queries.row(next_query).as_vector();
            slot.reset(query, dense_norms(query), next_query, pool_limit, //
                       u64_t(next_query) * 0x9E3779B97F4A7C15ull + slot_index + 1);
            ++next_query;
            node_id_t seed = slot.random_generator.bounded(size());
            slot.visited.set(seed);
            status_t emitted = emit_(context, slot_index, seed, dense_io_purpose_t::score_k);
            if (emitted != status_t::success_k) return emitted;
        }
        return status_t::success_k;
    }

    /** @brief Routes one completed read back into its slot & advances the traversal.
     *      Score completions are deferred - only their leased buffer index is stashed -
     *      so the quad kernel scores four arena-resident records per invocation. */
    inline status_t consume_(context_t& context, dense_io_completion_t completion) noexcept {
        dense_io_tag_t tag = dense_io_tag_t::unpack(completion.tag);
        slot_t& slot = context.slots[tag.slot];
        slot.inflight_reads--;
        if (completion.result < 0) return context.release_buffer(tag.buffer), status_t::device_error_k;

        if (static_cast<dense_io_purpose_t>(tag.purpose) == dense_io_purpose_t::score_k) {
            slot.pending_scores.try_push_back(tag.buffer);
            if (slot.pending_scores.full()) flush_scores_(context, slot);
        }
        else {
            dense_nvme_record_t record {context.engine.buffer(tag.buffer)};
            status_t expanded = expand_(context, tag.slot, record);
            context.release_buffer(tag.buffer);
            if (expanded != status_t::success_k) return expanded;
        }
        if (!slot.inflight_reads) {
            if (slot.pending_scores.size()) flush_scores_(context, slot); // The hop's short tail.
            advance_(context, tag.slot);
        }
        return status_t::success_k;
    }

    /** @brief Walks an expanded @p record's neighbor list & fetches every unvisited id
     *      for scoring - the fan-out half of one hop. */
    inline status_t expand_(context_t& context, dense_traversal_slot_id_t slot_index,
                            dense_nvme_record_t record) noexcept {
        slot_t& slot = context.slots[slot_index];
        context.expanded_members++;
        node_id_t const* neighbor_ids = record.neighbor_ids();
        u32_t neighbors_count = record.neighbors_count();
        for (small_size_t i = 0; i != neighbors_count; ++i) {
            node_id_t neighbor = neighbor_ids[i];
            if (neighbor == dense_node_id_unknown_k || slot.visited.set(neighbor)) continue;
            status_t emitted = emit_(context, slot_index, neighbor, dense_io_purpose_t::score_k);
            if (emitted != status_t::success_k) return emitted;
        }
        return status_t::success_k;
    }

    /** @brief Scores up to four deferred completions of one slot through a single quad
     *      kernel, merges them with one batched top-k insert whose accept mask gates the
     *      frontier, and releases their staging buffers. Short tails replicate lane 0 so
     *      the kernel always streams four valid lanes; only real lanes reach the pool or
     *      the telemetry. */
    inline void flush_scores_(context_t& context, slot_t& slot) noexcept {
        small_size_t fill = static_cast<small_size_t>(slot.pending_scores.size());
        std::array<vector_view_t, 4> lane_vectors;
        std::array<vector_norms_t, 4> lane_norms;
        std::array<node_id_t, 4> lane_nodes;
        for (small_size_t lane = 0; lane != 4; ++lane) {
            dense_nvme_record_t record {
                context.engine.buffer(slot.pending_scores[min_of<small_size_t>(lane, fill - 1)])};
            lane_vectors[lane] = vector_view_t {record.vector<scalar_t, vector_norms_t>(), dimensions_};
            lane_norms[lane] = record.norms<vector_norms_t>();
            lane_nodes[lane] = record.self_id();
        }
        std::array<distance_t, 4> scores = distances(slot.query, slot.query_norms, lane_vectors, lane_norms);
        context.computed_distances += fill;
        for (small_size_t lane = 0; lane != fill; ++lane)
            if (slot.results.insert(dense_scored_node_t {lane_nodes[lane], scores[lane]}, slot.pool_limit))
                slot.frontier.insert_reserved(dense_scored_node_t {lane_nodes[lane], scores[lane]});
        for (small_size_t lane = 0; lane != fill; ++lane) context.release_buffer(slot.pending_scores[lane]);
        slot.pending_scores.clear();
    }

    /** @brief With nothing in flight, either expands the most promising frontier node
     *      or declares convergence - the classic best-first termination: the closest
     *      unexpanded candidate is already further than the worst kept result. */
    inline void advance_(context_t& context, dense_traversal_slot_id_t slot_index) noexcept {
        slot_t& slot = context.slots[slot_index];
        while (slot.frontier.size()) {
            dense_scored_node_t best = slot.frontier.top();
            bool results_full = slot.results.size() == slot.pool_limit;
            if (results_full && best.distance > slot.results.top().distance) break;
            slot.frontier.pop();
            status_t emitted = emit_(context, slot_index, best.node_id, dense_io_purpose_t::expand_k);
            if (emitted != status_t::success_k) break;
            return;
        }
        slot.stage = dense_nvme_stage_t::finished_k;
    }

    /** @brief Copies every converged slot's ascending results into its output row &
     *      recycles the slot - the retirement half of the pump loop. */
    inline void harvest_slots_(context_t& context, nk::matrix_span<dense_scored_node_t> results,
                               nk::vector_span<small_size_t> result_counts, small_size_t wanted,
                               small_size_t& finished_queries) noexcept {
        for (dense_traversal_slot_id_t slot_index = 0; slot_index != context.slots.size(); ++slot_index) {
            slot_t& slot = context.slots[slot_index];
            if (slot.stage != dense_nvme_stage_t::finished_k) continue;
            small_size_t kept = static_cast<small_size_t>(min_of<std::size_t>(slot.results.size(), wanted));
            for (small_size_t i = 0; i != kept; ++i) results(slot.query_index, i) = slot.results[i];
            result_counts[slot.query_index] = kept;
            slot.stage = dense_nvme_stage_t::idle_k;
            ++finished_queries;
        }
    }

    offsets_tensor_t offsets_;
    small_size_t dimensions_ = 0;
    small_size_t connectivity_ceiling_ = 0;
    std::size_t record_capacity_ = 0;
#if !defined(_WIN32)
    int data_descriptor_ = -1;
#endif
};

#pragma endregion Index

} // namespace usearch
} // namespace unum
