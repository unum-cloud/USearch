/**
 *  @file include/usearch/utilities/nvme_device.hpp
 *  @author Ash Vardanian
 *  @date July 22, 2026
 *  @brief Host-side vocabulary for reading block storage: the media, the I/O engines that read
 *      them, and the executor that drives one engine per thread.
 *
 *  A walk over storage is a chain of dependent reads, each bounded by device latency, so throughput
 *  comes from keeping many independent reads in flight. Engines never decide what to read: they
 *  take requests, submit them through whatever asynchronous interface the platform offers, and hand
 *  back completions, so the walks above them stay pure state machines.
 *
 *  @section nvme_device_engines I/O Engines
 *
 *  Every engine satisfies the same @c io_engine concept and is named after the exact facility it
 *  drives:
 *
 *  - @c blocking_files_engine: blocking @c pread or @c ReadFile, the portable reference & macOS
 *    path,
 *  - @c linux_aio_files_engine: @c io_submit / @c io_getevents, for io_uring-gated clouds,
 *  - @c linux_uring_files_engine: io_uring reads, over registered buffers and polled queues on
 *    request,
 *  - @c linux_uring_nvme_engine: NVMe passthrough via @c IORING_OP_URING_CMD,
 *  - @c windows_ioring_files_engine: I/O Rings on Windows 11 22H2 & newer,
 *  - @c linux_spdk_engine: the SPDK user-space driver, in builds with @c USEARCH_WITH_LINUX_SPDK.
 *
 *  Engines own their staging buffers, since buffer registration is an engine-side resource on every
 *  ring API, and requests address buffers and media only by index. Requests carry an offset, a
 *  length, a medium ID, a buffer ID and an opaque context; completions return the buffer, a signed
 *  result and the context. The buffer is the cookie every kernel interface echoes, and an engine
 *  keeps each buffer's context beside it. No pointers cross the contract in either direction.
 *  The Linux ring engines mirror the kernel ABI they use, so they build against any headers, while
 *  runtime degrades gracefully on older kernels through setup fallbacks.
 *
 *  @section nvme_device_alignment Platform Alignment
 *
 *  The engine contract is the intersection of four asynchronous-read APIs. Each bullet is one
 *  contract verb, with the platforms in a fixed order: io_uring, then Linux AIO, then Windows
 *  IoRing, then the portable and macOS path.
 *
 *  - @b Open: @c io_uring_setup plus the ring mappings; @c io_setup for AIO; @c CreateIoRing on
 *    Windows; nothing to create portably.
 *  - @b Reserve: @c IORING_REGISTER_BUFFERS; no AIO equivalent; @c BuildIoRingRegisterBuffers;
 *    none portably.
 *  - @b Attach: @c IORING_REGISTER_FILES; no AIO equivalent; @c BuildIoRingRegisterFileHandles;
 *    none portably.
 *  - @b Enqueue: an SQE written in the shared ring; an @c iocb in an engine array;
 *    @c BuildIoRingReadFile builds into the ring; portably a staged request.
 *  - @b Submit: one @c io_uring_enter per batch; @c io_submit; @c SubmitIoRing; portably a loop
 *    of @c pread calls.
 *  - @b Poll: the CQ ring drained in user space; @c io_getevents; @c PopIoRingCompletion, also
 *    syscall-free; portably inline with submission.
 *  - @b Cookie: the buffer ID, as @c user_data on the SQE and CQE; @c aio_data echoed in
 *    @c io_event.data; IoRing's @c UserData; portably staged alongside.
 *  - @b Probe: setup-flag @c EINVAL fallback chains, so a refused ring falls to AIO and a refused
 *    AIO context to blocking reads; @c IsIoRingOpSupported; the portable path is the fallback.
 *
 *  A read consumes the SQE fields @c opcode, @c flags, @c fd, @c off, @c addr, @c len and
 *  @c user_data, plus @c buf_index for @c READ_FIXED, where @c fd becomes a registered-file index
 *  under @c IOSQE_FIXED_FILE, and @c cmd_op with the 72-byte inline @c nvme_uring_cmd for
 *  passthrough. A CQE returns only @c user_data, @c res and @c flags. Everything else in those
 *  64- or 128-byte slots stays zero, which is why the request below carries exactly four words.
 *
 *  @section nvme_device_emulation R&D Emulation
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
 *  echo 2     > poll_queues      # for polled rings & fio --hipri
 *  echo 1     > power            # creates /dev/nvme0 (named after this directory!)
 *  @endcode
 *
 *  Two caveats measured on 6.11: the block device is named after the configfs
 *  directory, not @c nullb0; and @c completion_nsec is honored accurately on IRQ
 *  queues but only coarsely on poll queues, so calibrate polled-path latency
 *  against an IRQ-mode run of the same instance.
 */
#pragma once

#include <cassert> // `assert`
#include <cerrno>  // `errno`, `EINTR`, `EINVAL`, `EIO`
#include <cstddef> // `std::size_t`
#include <cstdint> // `std::uintptr_t`
#include <cstdio>  // `std::snprintf`, for the sysfs path of a medium's disk
#include <cstring> // `std::memcpy`, `std::memset`

#include <bit>         // `std::countr_zero`
#include <charconv>    // `std::from_chars`, for sysfs attributes
#include <concepts>    // `std::same_as`
#include <limits>      // `std::numeric_limits`, for the longest read a result counts
#include <mutex>       // `std::unique_lock`
#include <new>         // `std::nothrow`
#include <span>        // `std::span`
#include <string_view> // `std::string_view`
#include <type_traits> // `std::conditional_t`, `std::is_same_v`, `std::is_trivially_copyable_v`
#include <utility>     // `std::exchange`, `std::swap`, `std::move`
#include <variant>     // `std::variant`, `std::visit`, `std::get_if`, `std::in_place_index`

#include <forkunion/forkunion.h> // `fu_pool_new`, `fu_pool_spawn`, `fu_pool_delete`
#include <forkunion/atomics.hpp> // `fu::standard_atomic_ref` - the SPDK claim word
#include <forkunion/types.hpp>   // `fu::wait_capped_t`, `fu::spin_mutex_t`

#include "usearch/utilities/blocking_files.hpp" // `native_file_t`, `open_native_file`, `read_native_file`
#include "usearch/utilities/ram_buffers.hpp"    // `growing_buffer`
#include "usearch/utilities/ram_device.hpp"     // `forkunion_executor_t`, `forkunion_allocator_t`
#include "usearch/utilities/types.hpp"          // `status_t`, `expected`, `min_of`, `max_of`

#if USEARCH_OS_WINDOWS_
#include <windows.h> // `CreateFileA`, `ReadFile`, `GetOverlappedResult`, `GetFileInformationByHandleEx`
#else
#include <fcntl.h>    // `open`, `fcntl`, `O_DIRECT`, `F_NOCACHE`
#include <sys/stat.h> // `fstat`, `statx`, `S_ISBLK`
#include <unistd.h>   // `pread`, `close`, `syscall`
#endif

#if USEARCH_OS_LINUX_
#include <dirent.h>        // `fdopendir`, `readdir`, for a disk's hardware queues and an ng node's namespaces
#include <sys/ioctl.h>     // `ioctl`, `_IO`, `_IOWR`
#include <sys/mman.h>      // `mmap`, `munmap`
#include <sys/syscall.h>   // `SYS_io_setup`, `SYS_statx`
#include <sys/sysmacros.h> // `major`, `minor`
#include <sys/uio.h>       // `struct iovec`
#endif

#if USEARCH_WITH_LINUX_AIO
#include <linux/aio_abi.h> // `aio_context_t`, `struct iocb`, `struct io_event`, `IOCB_CMD_PREAD`
#endif

#if USEARCH_WITH_WINDOWS_IORING
#include <ioringapi.h> // `CreateIoRing`, `BuildIoRingReadFile`, `SubmitIoRing`, `PopIoRingCompletion`
#endif

#if USEARCH_WITH_LINUX_SPDK
#include <sched.h>         // `sched_getaffinity`, `sched_setaffinity`, undoing the EAL's pinning
#include <spdk/env.h>      // `spdk_env_init`, `spdk_env_fini`, `spdk_dma_malloc_socket`, `spdk_pci_device_get_addr`
#include <spdk/env_dpdk.h> // `spdk_env_dpdk_external_init`
#include <spdk/nvme.h>     // `spdk_nvme_connect`, `spdk_nvme_ns_cmd_read`, `spdk_nvme_qpair_process_completions`
#endif

namespace unum::usearch::utilities {

#pragma region Requests and Completions

/** Bytes every staging buffer aligns to: a page, the strictest memory alignment direct I/O asks of
 *  any mainstream device, so one buffer never straddles two pages. */
inline constexpr std::size_t io_alignment_k = 4096;

/** The block a medium reports when nothing finer is known: the finest NVMe logical block. */
inline constexpr u32_t io_block_bytes_default_k = 512;

/** Index of one staging buffer an engine owns: requests name their destination by it. */
using io_buffer_id_t = u32_t;

/** Index into the media an engine's @c attach received, so one engine serves every shard a call
 *  reads. */
using io_medium_id_t = u32_t;

/** The longest read a request may ask for: the most bytes a completion's signed result counts. */
inline constexpr u32_t io_max_length_k = static_cast<u32_t>(std::numeric_limits<i32_t>::max());

/** The NVM command set's Read opcode. */
inline constexpr u8_t nvme_read_opcode_k = 0x02;

/** Whether @c submit returns with what has already completed, or holds for at least one
 *  completion. */
enum class io_wait_t : u32_t {

    /** Submit, and return without sleeping. */
    nonblocking_k,

    /** Submit, then hold until a completion arrives, unless nothing is in flight. */
    blocking_k,
};

/** One read an engine performs: where, how much, into which buffer, and the context to hand back.
 *  The destination address never appears; the engine derives it from @c buffer. */
struct io_request_t {

    /** Byte offset on the medium, a multiple of its block. */
    u64_t offset = 0;

    /** Bytes to read, a multiple of the medium's block, at most @c io_max_length_k. */
    u32_t length = 0;

    /** Which attached medium the read targets. */
    io_medium_id_t medium = 0;

    /** The staging buffer the bytes arrive in, serving no other read until the completion is
     *  polled. */
    io_buffer_id_t buffer = 0;

    /** Opaque to engines, handed back verbatim in the completion. */
    u64_t context = 0;
};

/** One finished read, as @c poll_completions hands it back. */
struct io_completion_t {

    /** The buffer of the request it answers. */
    io_buffer_id_t buffer = 0;

    /** Bytes transferred, or a negated @c errno when the read failed. */
    i32_t result = 0;

    /** The context of the request it answers. */
    u64_t context = 0;
};

static_assert(sizeof(io_request_t) == 4 * sizeof(u64_t) && std::is_trivially_copyable_v<io_request_t>,
              "Requests are carved without construction and copied as four words");
static_assert(sizeof(io_completion_t) == 2 * sizeof(u64_t) && std::is_trivially_copyable_v<io_completion_t>,
              "Completions are carved without construction and copied as two words");

/** The transport an engine drives, and the one a configuration asks for. */
enum class io_transport_t : u32_t {

    /** The first kernel files engine whose setup the host admits: io_uring, then AIO, then blocking
     *  reads. Any other transport is named exactly: its engine opens, or the open says why not. */
    automatic_k,

    /** @c pread or @c ReadFile on the calling thread. */
    blocking_k,

    /** io_uring reads on files and block devices. */
    linux_uring_k,

    /** Linux AIO over @c io_submit and @c io_getevents. */
    linux_aio_k,

    /** io_uring NVMe passthrough on namespace character devices, from Linux 5.19. */
    linux_uring_nvme_k,

    /** The SPDK user-space driver, over controllers named by transport IDs. */
    linux_spdk_k,

    /** I/O Rings on Windows 11 22H2 and newer. */
    windows_ioring_k,
};

/** Whether io_uring reads arrive in buffers registered once or at plain addresses the kernel pins
 *  on every read. */
enum class linux_uring_buffer_registration_t : u32_t {

    /** Plain addresses, which the kernel pins on every read. */
    plain_k,

    /** Buffers registered once, roughly an 11% throughput gain at NVMe rates. */
    fixed_k,
};

/** Whether io_uring learns of completions by interrupt or by polling the device's queues. */
enum class linux_uring_reaping_t : u8_t {

    /** Interrupt-driven completions. */
    interrupted_k,

    /** @c IORING_SETUP_IOPOLL, which needs direct I/O and poll queues, about 21% over
     *  interrupts. */
    polled_k,
};

/** Keeps the first failure any thread reports into @p first, through the CPU class's reference. */
template <template <typename> class atomic_ref_template_>
void keep_first_failure(status_t& first, status_t status) noexcept {
    if (succeeded(status)) return;
    status_t unset = status_t::success_k;
    atomic_ref_template_<status_t>(first).compare_exchange_strong(unset, status, std::memory_order_relaxed,
                                                                  std::memory_order_relaxed);
}

#pragma endregion Requests and Completions

#pragma region Media

/** What the block queue under a medium admits, read from sysfs when it opens; zeros where no
 *  multi-queue disk backs it, as on tmpfs, overlays and bio-based device-mapper targets. */
struct io_device_queues_t {

    /** The whole disk's major and minor, packed as the kernel packs them, so parts on one disk
     *  count its queues once; zero when no disk backs the medium. */
    u32_t device = 0;

    /** Requests one hardware queue holds: the disk's @c queue/nr_requests. */
    u16_t tags = 0;

    /** Hardware queues the disk spreads CPUs over: the entries of its @c mq/. */
    u16_t hardware_queues = 0;

    /** @c polled_k when the disk has poll queues and the medium reads direct. */
    linux_uring_reaping_t reaping = linux_uring_reaping_t::interrupted_k;

    /** The most KiB one request carries before the disk splits it: the disk's
     *  @c queue/max_sectors_kb, or the controller's maximum transfer; zero when unknown. */
    u16_t request_kibibytes = 0;
};

/** A file or block device open for direct reads, closed when it dies: @c O_DIRECT where the
 *  filesystem admits it, @c F_NOCACHE on Apple, and @c FILE_FLAG_NO_BUFFERING on Windows, all
 *  through @ref open_native_file. */
struct file_medium_t {

    /** The open descriptor, @c invalid_native_file_k once moved out. */
    native_file_t descriptor = invalid_native_file_k;

    /** Where the part's byte zero sits in the file or on the device. */
    u64_t byte_base = 0;

    /** The granularity of a direct read's offset and length on it. */
    u32_t block_bytes = 0;

    /** The block queue of the disk under it, which a call's queue depth and reaping derive from. */
    io_device_queues_t queues;

    file_medium_t() noexcept = default;
    file_medium_t(file_medium_t&& other) noexcept
        : descriptor(std::exchange(other.descriptor, invalid_native_file_k)),
          byte_base(std::exchange(other.byte_base, 0u)), block_bytes(std::exchange(other.block_bytes, 0u)),
          queues(std::exchange(other.queues, {})) {}
    file_medium_t& operator=(file_medium_t&& other) noexcept {
        std::swap(descriptor, other.descriptor);
        std::swap(byte_base, other.byte_base);
        std::swap(block_bytes, other.block_bytes);
        std::swap(queues, other.queues);
        return *this;
    }
    file_medium_t(file_medium_t const&) = delete;
    file_medium_t& operator=(file_medium_t const&) = delete;
    ~file_medium_t() noexcept {
        if (descriptor != invalid_native_file_k) [[maybe_unused]]
            status_t const closed = close_native_file(descriptor);
    }

    /**
     *  Opens @p path read-only, a file or a device in any form @ref file_address_t lists, its base
     *  a multiple of the block its direct reads need; the NUL terminator the OS demands is one
     *  transient allocation.
     *
     *  @par Statuses:
     *  - @c invalid_argument_k @p path names no location, or its base or namespace is unreadable.
     *  - @c invalid_argument_k The base is not a multiple of the medium's block.
     */
    static expected<file_medium_t> open(std::string_view path) noexcept {
        auto const [parsed, parse_status] = parse_file_address(path);
        if (failed(parse_status)) return {file_medium_t {}, parse_status};
        growing_buffer<char> terminated {global_allocator};
        std::size_t const length = parsed.location.size();
        if (failed(terminated.resize_uninitialized(length + 1))) return {file_medium_t {}, status_t::bad_alloc_k};
        terminated[parsed.location.copy(terminated.data(), length)] = '\0';
        auto [medium, open_status] = open_(terminated.data());
        if (failed(open_status)) return {std::move(medium), open_status};
        if (parsed.byte_base % medium.block_bytes) return {std::move(medium), status_t::invalid_argument_k};
        medium.byte_base = parsed.byte_base;
        return {std::move(medium), status_t::success_k};
    }

    /**
     *  Fills @p bytes from @p offset bytes past the part's start, one block-aligned blocking read
     *  through a staging buffer of its own, for the headers a view reads once.
     *
     *  @par Statuses:
     *  - @c overflow_k The blocks covering @p bytes span more than one read can move.
     *  - @c io_k The read ended before the last byte of @p bytes.
     */
    status_t read_at(u64_t offset, bytes_span_t bytes) const noexcept;

  private:

    /** Opens the terminated @p path, as @ref open promises. */
    static expected<file_medium_t> open_(char const* path) noexcept;
};

static_assert(sizeof(io_device_queues_t) == 12 && sizeof(file_medium_t) == 32, "Media stay a 32-byte hot array");

#if USEARCH_OS_WINDOWS_

inline expected<file_medium_t> file_medium_t::open_(char const* path) noexcept {
    file_medium_t medium;
    medium.descriptor = open_native_file(path, file_access_t::read_only_k, file_buffering_t::unbuffered_k);
    if (medium.descriptor == invalid_native_file_k) return {std::move(medium), status_t::io_k};
    FILE_STORAGE_INFO storage {};
    medium.block_bytes = ::GetFileInformationByHandleEx(medium.descriptor, FileStorageInfo, &storage,
                                                        sizeof(storage)) &&
                                 storage.LogicalBytesPerSector
                             ? static_cast<u32_t>(storage.LogicalBytesPerSector)
                             : io_block_bytes_default_k;
    return {std::move(medium), status_t::success_k};
}

inline i32_t nvme_namespace_read_(HANDLE, bytes_span_t, u64_t, u32_t) noexcept { return -ENODEV; }

/** Bytes @p medium spans past its base. */
inline u64_t medium_bytes_of(file_medium_t const& medium) noexcept {
    u64_t const bytes = native_file_capacity(medium.descriptor).value;
    return bytes - min_of(bytes, medium.byte_base);
}

/** Whether reads on @p medium skip the page cache, as every handle opened unbuffered does. */
inline bool reads_directly(file_medium_t const&) noexcept { return true; }

/** The memory domain nearest the disk under @p medium, which only Linux reports. */
inline fu_memory_domain_id_t memory_domain_of(file_medium_t const&) noexcept { return no_memory_domain_k; }

#endif

#if USEARCH_OS_LINUX_

#pragma region Kernel

/** @c IOSQE_FIXED_FILE: an entry's @c fd indexes the registered file table. */
enum linux_uring_entry_flag_t : u8_t {
    linux_uring_entry_fixed_file_k = 1u << 0,
};

/** @c io_sqring_offsets: where the submission ring's words sit in its mapping. */
struct linux_uring_submission_offsets_t {
    u32_t head_offset;
    u32_t tail_offset;
    u32_t ring_mask_offset;
    u32_t ring_entries_offset;
    u32_t flags_offset;
    u32_t dropped_offset;
    u32_t array_offset;
    u32_t reserved;
    u64_t reserved2;
};

/** @c io_cqring_offsets: where the completion ring's words sit in its mapping. */
struct linux_uring_completion_offsets_t {
    u32_t head_offset;
    u32_t tail_offset;
    u32_t ring_mask_offset;
    u32_t ring_entries_offset;
    u32_t overflow_offset;
    u32_t cqes_offset;
    u32_t flags_offset;
    u32_t reserved;
    u64_t reserved2;
};

/** @c io_uring_params: what setup is asked for and what the kernel answers. */
struct linux_uring_params_t {
    u32_t sq_entries;
    u32_t cq_entries;
    u32_t setup_flags;
    u32_t sq_thread_cpu;
    u32_t sq_thread_idle;
    u32_t features;
    u32_t wq_fd;
    u32_t reserved[3];
    linux_uring_submission_offsets_t sq_off;
    linux_uring_completion_offsets_t cq_off;
};
static_assert(sizeof(linux_uring_params_t) == 120, "the kernel reads a 120-byte io_uring_params");

/** @c io_uring_sqe in its 64-byte form, the one a block-layer read takes. */
struct linux_uring_sqe_t {
    u8_t opcode;
    u8_t entry_flags;
    u16_t ioprio;
    i32_t fd;
    u64_t offset;
    u64_t addr;
    u32_t length;
    u32_t rw_flags;
    u64_t user_data;
    u16_t buf_index;
    u16_t personality;
    i32_t file_index;
    u64_t addr3;
    u64_t padding2;
};
static_assert(sizeof(linux_uring_sqe_t) == 64, "the kernel reads 64-byte entries from a plain ring");

/** @c io_uring_sqe in its 128-byte form, the one an NVMe passthrough ring takes, whose command
 *  fills the last 80 bytes. */
struct linux_uring_sqe128_t {
    u8_t opcode;
    u8_t entry_flags;
    u16_t ioprio;
    i32_t fd;
    u32_t cmd_op;
    u32_t padding;
    u64_t addr;
    u32_t length;
    u32_t uring_cmd_flags;
    u64_t user_data;
    u16_t buf_index;
    u16_t personality;
    i32_t file_index;
    u8_t cmd[80];
};
static_assert(sizeof(linux_uring_sqe128_t) == 128, "the kernel reads 128-byte entries from an SQE128 ring");

/** @c io_uring_cqe in its 16-byte form; a CQE32 ring pads each to 32. */
struct linux_uring_cqe_t {
    u64_t user_data;
    i32_t res;
    u32_t completion_flags;
};
static_assert(sizeof(linux_uring_cqe_t) == 16, "the kernel writes 16-byte completions to a plain ring");

/** @c io_uring_rsrc_update: one slot of a registration update. */
struct linux_uring_rsrc_update_t {
    u32_t offset;
    u32_t reserved;
    u64_t update_value;
};
static_assert(sizeof(linux_uring_rsrc_update_t) == 16, "the kernel reads a 16-byte io_uring_rsrc_update");

/** @c nvme_passthru_cmd: the NVMe command the @c NVME_IOCTL_IO_CMD ioctl takes. */
struct linux_nvme_passthru_cmd_t {
    u8_t opcode;
    u8_t command_flags;
    u16_t reserved;
    u32_t nsid;
    u32_t cdw2, cdw3;
    u64_t metadata;
    u64_t addr;
    u32_t metadata_length;
    u32_t data_length;
    u32_t cdw10, cdw11, cdw12, cdw13, cdw14, cdw15;
    u32_t timeout_ms;
    u32_t result;
};
static_assert(sizeof(linux_nvme_passthru_cmd_t) == 72, "the kernel reads a 72-byte NVMe ioctl command");

/** @c nvme_uring_cmd: the NVMe command an @c IORING_OP_URING_CMD entry carries in @c cmd, from
 *  Linux 5.19. */
struct linux_nvme_uring_cmd_t {
    u8_t opcode;
    u8_t command_flags;
    u16_t reserved;
    u32_t nsid;
    u32_t cdw2, cdw3;
    u64_t metadata;
    u64_t addr;
    u32_t metadata_length;
    u32_t data_length;
    u32_t cdw10, cdw11, cdw12, cdw13, cdw14, cdw15;
    u32_t timeout_ms;
    u32_t reserved2;
};
static_assert(sizeof(linux_nvme_uring_cmd_t) == 72, "the kernel reads a 72-byte NVMe passthrough command");

/** @c NVME_IOCTL_ID: the ioctl answering a namespace node's NSID. */
enum linux_nvme_ioctl_t : unsigned long {
    linux_nvme_ioctl_id_k = _IO('N', 0x40),
};

/** @c statx_timestamp. */
struct linux_statx_timestamp_t {
    i64_t tv_sec;
    u32_t tv_nsec;
    i32_t reserved;
};

/** @c statx, with the unbuffered-read alignments Linux 6.1 added. */
struct linux_statx_t {
    u32_t mask;
    u32_t blksize;
    u64_t attributes;
    u32_t nlink;
    u32_t uid;
    u32_t gid;
    u16_t mode;
    u16_t spare0;
    u64_t ino;
    u64_t size;
    u64_t blocks;
    u64_t attributes_mask;
    linux_statx_timestamp_t atime;
    linux_statx_timestamp_t btime;
    linux_statx_timestamp_t ctime;
    linux_statx_timestamp_t mtime;
    u32_t rdev_major;
    u32_t rdev_minor;
    u32_t dev_major;
    u32_t dev_minor;
    u64_t mnt_id;
    u32_t dio_mem_align;
    u32_t dio_offset_align;
    u64_t spare3[12];
};
static_assert(sizeof(linux_statx_t) == 256, "the kernel writes a 256-byte statx");

#pragma endregion Kernel

/** The direct-I/O offset alignment the filesystem reports for @p descriptor, zero when it reports
 *  none, as kernels before 6.1 do. */
inline u32_t direct_offset_alignment_(int descriptor) noexcept {
    u32_t const dioalign = 0x2000; // STATX_DIOALIGN
    linux_statx_t description {};
    if (::syscall(SYS_statx, descriptor, "", AT_EMPTY_PATH, dioalign, &description) != 0) return 0;
    return description.mask & dioalign ? description.dio_offset_align : 0;
}

#endif // USEARCH_OS_LINUX_

#if USEARCH_OS_LINUX_

/** The block queue under a medium and the logical block its disk addresses. */
struct io_block_queue_t {

    /** What the disk's block queue admits. */
    io_device_queues_t queues;

    /** The disk's @c queue/logical_block_size, zero when no disk backs the medium. */
    u32_t logical_block_bytes = 0;
};

/** The unsigned decimal leading the sysfs attribute @p name under @p directory, and in @p minor the
 *  one past its colon, as @c dev spells them; zero when the attribute or the part is missing. */
inline u64_t sysfs_number_(int directory, char const* name, u64_t* minor = nullptr) noexcept {
    int const attribute = ::openat(directory, name, O_RDONLY | O_CLOEXEC);
    if (attribute < 0) return 0;
    // Two 64-bit decimals and their separators, as `dev` spells them, the longest attribute here.
    char text[2 * (std::numeric_limits<u64_t>::digits10 + 2)];
    ssize_t const length = ::read(attribute, text, sizeof(text));
    ::close(attribute);
    u64_t number = 0;
    if (length <= 0) return 0;
    std::from_chars_result const parsed = std::from_chars(text, text + length, number);
    if (minor && parsed.ptr != text + length && *parsed.ptr == ':')
        std::from_chars(parsed.ptr + 1, text + length, *minor);
    return parsed.ec == std::errc {} ? number : 0;
}

/** The memory domain nearest the PCI function behind the disk at @p disk, as its controller's
 *  @c device/address names it; @c no_memory_domain_k off PCI or tied to no domain. */
inline fu_memory_domain_id_t sysfs_memory_domain_(int disk) noexcept {
    int const attribute = ::openat(disk, "device/address", O_RDONLY | O_CLOEXEC);
    if (attribute < 0) return no_memory_domain_k;
    char bus_id[32] = {};
    ssize_t const length = ::read(attribute, bus_id, sizeof(bus_id) - 1);
    ::close(attribute);
    if (length <= 0) return no_memory_domain_k;
    if (bus_id[length - 1] == '\n') bus_id[length - 1] = 0;
    fu_memory_domain_id_t domain = no_memory_domain_k;
    return fu_memory_domain_id_of_pci(bus_id, &domain) == fu_success_k ? domain : no_memory_domain_k;
}

/** Hardware queues the disk at @p disk spreads CPUs over: the entries of its @c mq/. */
inline u32_t sysfs_hardware_queues_(int disk) noexcept {
    int const listed = ::openat(disk, "mq", O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (listed < 0) return 0;
    DIR* const entries = ::fdopendir(listed);
    if (!entries) return ::close(listed), 0u;
    u32_t queues = 0;
    for (struct dirent const* entry = ::readdir(entries); entry; entry = ::readdir(entries))
        queues += entry->d_name[0] != '.';
    ::closedir(entries);
    return queues;
}

/** The sysfs directory of the namespace disk behind the generic char node @p descriptor, whose
 *  controller directory is @p controller: the entry whose @c nsid matches @c NVME_IOCTL_ID. */
inline int sysfs_namespace_disk_(int descriptor, int controller) noexcept {
    int const identifier = ::ioctl(descriptor, linux_nvme_ioctl_id_k);
    if (identifier <= 0) return -1;
    // The listing takes a descriptor of its own, which `closedir` closes.
    int const listed = ::fcntl(controller, F_DUPFD_CLOEXEC, 0);
    DIR* const entries = listed >= 0 ? ::fdopendir(listed) : nullptr;
    if (!entries) {
        if (listed >= 0) ::close(listed);
        return -1;
    }
    int disk = -1;
    for (struct dirent const* entry = ::readdir(entries); entry && disk < 0; entry = ::readdir(entries)) {
        if (entry->d_name[0] == '.') continue;
        int const candidate = ::openat(controller, entry->d_name, O_RDONLY | O_DIRECTORY | O_CLOEXEC);
        if (candidate < 0) continue;
        // The char node's own entry carries no `nsid`, and neither do the controller's attributes.
        if (sysfs_number_(candidate, "nsid") == static_cast<u64_t>(identifier) &&
            ::faccessat(candidate, "queue", F_OK, 0) == 0)
            disk = candidate;
        else ::close(candidate);
    }
    ::closedir(entries);
    return disk;
}

/** The sysfs directory of the whole disk under @p descriptor, described by @p description: a block
 *  node's own disk or its partition's parent, a file's disk through @c st_dev, or the namespace
 *  disk of a generic char node; -1 where none backs it. */
inline int sysfs_disk_of_(int descriptor, struct stat const& description) noexcept {
    bool const character = S_ISCHR(description.st_mode);
    dev_t const device = S_ISBLK(description.st_mode) || character ? description.st_rdev : description.st_dev;
    // The longest path the two roots spell with 32-bit majors and minors.
    char path[sizeof("/sys/dev/block/4294967295:4294967295/device")];
    std::snprintf(path, sizeof(path), character ? "/sys/dev/char/%u:%u/device" : "/sys/dev/block/%u:%u",
                  static_cast<unsigned>(major(device)), static_cast<unsigned>(minor(device)));
    int disk = ::open(path, O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (disk >= 0 && character) {
        int const controller = std::exchange(disk, -1);
        disk = sysfs_namespace_disk_(descriptor, controller);
        ::close(controller);
    }
    else if (disk >= 0 && ::faccessat(disk, "partition", F_OK, 0) == 0) {
        int const partition = std::exchange(disk, -1);
        disk = ::openat(partition, "..", O_RDONLY | O_DIRECTORY | O_CLOEXEC);
        ::close(partition);
    }
    return disk;
}

/** Reads the block queue of the whole disk under @p descriptor, described by @p description, from
 *  sysfs. @p refused is the @c errno of the @c O_DIRECT open, zero when it held, and reaping polls
 *  only on direct reads. */
inline io_block_queue_t io_block_queue_of_(int descriptor, struct stat const& description, int refused) noexcept {
    io_block_queue_t queue;
    bool const character = S_ISCHR(description.st_mode);
    int const disk = sysfs_disk_of_(descriptor, description);
    if (disk < 0) return queue;
    queue.logical_block_bytes = static_cast<u32_t>(sysfs_number_(disk, "queue/logical_block_size"));
    u64_t disk_minor = 0;
    u64_t const disk_major = sysfs_number_(disk, "dev", &disk_minor);
    u64_t const tags = sysfs_number_(disk, "queue/nr_requests");
    u32_t const hardware_queues = sysfs_hardware_queues_(disk);
    bool const polls = sysfs_number_(disk, "queue/io_poll") == 1 && (!refused || character);
    if (hardware_queues && tags && disk_major) {
        // The kernel's own `dev_t` keeps the minor in the low 20 bits.
        queue.queues.device = static_cast<u32_t>(disk_major << 20) | static_cast<u32_t>(disk_minor);
        queue.queues.tags = static_cast<u16_t>(min_of<u64_t>(tags, std::numeric_limits<u16_t>::max()));
        queue.queues.hardware_queues = static_cast<u16_t>(
            min_of<u32_t>(hardware_queues, std::numeric_limits<u16_t>::max()));
        queue.queues.reaping = polls ? linux_uring_reaping_t::polled_k : linux_uring_reaping_t::interrupted_k;
        queue.queues.request_kibibytes = static_cast<u16_t>(
            min_of<u64_t>(sysfs_number_(disk, "queue/max_sectors_kb"), std::numeric_limits<u16_t>::max()));
    }
    ::close(disk);
    return queue;
}

inline expected<file_medium_t> file_medium_t::open_(char const* path) noexcept {
    file_medium_t medium;
    medium.descriptor = open_native_file(path, file_access_t::read_only_k, file_buffering_t::unbuffered_k);
    // Filesystems without direct I/O, like tmpfs, and ng nodes refuse the flag, not ignore it.
    int const refused = medium.descriptor == invalid_native_file_k ? errno : 0;
    if (refused == EINVAL)
        medium.descriptor = open_native_file(path, file_access_t::read_only_k, file_buffering_t::buffered_k);
    if (medium.descriptor == invalid_native_file_k) return {std::move(medium), status_t::io_k};
    struct stat description;
    if (::fstat(medium.descriptor, &description) != 0) return {std::move(medium), status_t::io_k};
    // The whole disk's queue: a node's own, a partition's or a file's parent, or an ng node's disk.
    io_block_queue_t const queue = io_block_queue_of_(medium.descriptor, description, refused);
    medium.queues = queue.queues;
    medium.block_bytes = direct_offset_alignment_(medium.descriptor);
    if (!medium.block_bytes) medium.block_bytes = queue.logical_block_bytes;
    if (!medium.block_bytes) medium.block_bytes = io_block_bytes_default_k;
    return {std::move(medium), status_t::success_k};
}

/** Reads @p destination at @p offset of the namespace behind the generic char node @p descriptor,
 *  one NVMe Read per memory page, which every controller's MDTS covers at least twice. */
inline i32_t nvme_namespace_read_(int descriptor, bytes_span_t destination, u64_t offset, u32_t block_bytes) noexcept {
    unsigned long const io_cmd = _IOWR('N', 0x43, linux_nvme_passthru_cmd_t); // NVME_IOCTL_IO_CMD
    int const identifier = ::ioctl(descriptor, linux_nvme_ioctl_id_k);
    if (identifier < 0) return -EINVAL;
    u32_t const step = max_of<u32_t>(static_cast<u32_t>(io_alignment_k), block_bytes);
    u32_t const length = static_cast<u32_t>(destination.size());
    for (u32_t moved = 0; moved != length;) {
        u32_t const bytes = min_of(step, length - moved);
        u64_t const first_block = (offset + moved) / block_bytes;
        linux_nvme_passthru_cmd_t command {};
        command.opcode = nvme_read_opcode_k, command.nsid = static_cast<u32_t>(identifier);
        command.addr = reinterpret_cast<std::uintptr_t>(destination.data() + moved), command.data_length = bytes;
        command.cdw10 = static_cast<u32_t>(first_block), command.cdw11 = static_cast<u32_t>(first_block >> 32);
        command.cdw12 = bytes / block_bytes - 1;
        if (::ioctl(descriptor, io_cmd, &command) != 0) return -EIO;
        moved += bytes;
    }
    return static_cast<i32_t>(length);
}

/** Bytes @p medium spans past its base: a file's length, a block device's, or the namespace's
 *  behind a generic char node, whose disk counts 512-byte sectors in sysfs whatever its block. */
inline u64_t medium_bytes_of(file_medium_t const& medium) noexcept {
    struct stat description;
    if (::fstat(medium.descriptor, &description) != 0) return 0;
    u64_t bytes = static_cast<u64_t>(description.st_size);
    if (S_ISBLK(description.st_mode)) bytes = block_device_bytes_(medium.descriptor);
    int const disk = S_ISCHR(description.st_mode) ? sysfs_disk_of_(medium.descriptor, description) : -1;
    if (disk >= 0) bytes = sysfs_number_(disk, "size") * 512, ::close(disk);
    return bytes - min_of(bytes, medium.byte_base);
}

/** Whether reads on @p medium skip the page cache: @c O_DIRECT held, or a generic char node, which
 *  has none, though it refuses the flag. */
inline bool reads_directly(file_medium_t const& medium) noexcept {
    struct stat description;
    int const flags = ::fcntl(medium.descriptor, F_GETFL);
    return (flags >= 0 && (flags & O_DIRECT)) ||
           (::fstat(medium.descriptor, &description) == 0 && S_ISCHR(description.st_mode));
}

/** The memory domain nearest the PCI function of the disk under @p medium, where its threads
 *  belong; @c no_memory_domain_k off PCI or where the kernel ties it to none. */
inline fu_memory_domain_id_t memory_domain_of(file_medium_t const& medium) noexcept {
    struct stat description;
    if (::fstat(medium.descriptor, &description) != 0) return no_memory_domain_k;
    int const disk = sysfs_disk_of_(medium.descriptor, description);
    if (disk < 0) return no_memory_domain_k;
    fu_memory_domain_id_t const domain = sysfs_memory_domain_(disk);
    ::close(disk);
    return domain;
}

#elif !USEARCH_OS_WINDOWS_

inline expected<file_medium_t> file_medium_t::open_(char const* path) noexcept {
    file_medium_t medium;
    medium.descriptor = open_native_file(path, file_access_t::read_only_k, file_buffering_t::unbuffered_k);
    if (medium.descriptor == invalid_native_file_k) return {std::move(medium), status_t::io_k};
    medium.block_bytes = io_block_bytes_default_k;
    return {std::move(medium), status_t::success_k};
}

inline i32_t nvme_namespace_read_(int, bytes_span_t, u64_t, u32_t) noexcept { return -ENODEV; }

/** Bytes @p medium spans past its base. */
inline u64_t medium_bytes_of(file_medium_t const& medium) noexcept {
    u64_t const bytes = native_file_capacity(medium.descriptor).value;
    return bytes - min_of(bytes, medium.byte_base);
}

/** Whether reads on @p medium skip the page cache; Apple's @c F_NOCACHE is a hint, and FreeBSD
 *  serves unaligned @c O_DIRECT reads from the cache, so they may not. */
inline bool reads_directly(file_medium_t const&) noexcept { return false; }

/** The memory domain nearest the disk under @p medium, which only Linux reports. */
inline fu_memory_domain_id_t memory_domain_of(file_medium_t const&) noexcept { return no_memory_domain_k; }

#endif // USEARCH_OS_LINUX_

/** The block every read on @p medium rounds its offset and length to. */
inline u32_t block_bytes_of(file_medium_t const& medium) noexcept { return medium.block_bytes; }

/** Where a part's byte zero sits on @p medium. */
inline u64_t byte_base_of(file_medium_t const& medium) noexcept { return medium.byte_base; }

/** What the block queue of the disk under @p medium admits. */
inline io_device_queues_t device_queues_of(file_medium_t const& medium) noexcept { return medium.queues; }

/** Whether @p path names a namespace by its SPDK transport ID or an @c spdk:// address, which only
 *  the SPDK engine reaches, rather than a file or a device node. */
inline bool names_transport_id(std::string_view path) noexcept {
    return path.starts_with("spdk://") || path.find("trtype:") != std::string_view::npos;
}

#if USEARCH_WITH_LINUX_SPDK

/** Where the SPDK environment takes its memory. */
enum class linux_spdk_memory_t : u32_t {

    /** Hugepages, which PCIe controllers read into by DMA. */
    hugepages_k,

    /** Plain pages and no PCIe scan, for fabrics like NVMe/TCP that copy every byte through a
     *  socket. */
    plain_pages_k,
};

/**
 *  @brief The SPDK environment and the controllers this process attached, shared by every view and
 *      engine that reads through them.
 *
 *  The environment starts once per process and never finishes, since DPDK cannot start twice, so
 *  every later @c start shares it, config and all. A controller counts its holders, each namespace
 *  a view reads and each engine's queue pair on it, and the last @c release detaches it. SPDK
 *  answers a transport ID with the controller still attached to it, failed or not, so a failed
 *  controller refuses new holders until its last one releases, and the next @c connect attaches
 *  whichever drive answers then. Every change waits on one claim word through the @p wait_type_
 *  its caller names, and none happens in a destructor. DPDK is not fork-safe, so a child process
 *  never touches the environment, and its multi-process mode is out of scope.
 */
class linux_spdk_process_t {
  public:

    /** How the environment starts. */
    struct config_t {

        /** Hugepages for PCIe controllers, or plain pages for fabrics. */
        linux_spdk_memory_t memory = linux_spdk_memory_t::hugepages_k;

        /** Megabytes of plain pages reserved at once; hugepage environments grow on demand. */
        std::size_t plain_megabytes = 256;
    };

    /** One controller this process attached, the transport ID it answered, and its holders. */
    struct connected_t {
        spdk_nvme_transport_id transport;
        spdk_nvme_ctrlr* controller;
        std::size_t holders;
    };

    /** Whether the environment runs, so engines may take DMA buffers and queue pairs from it. */
    static bool running() noexcept {
        return fu::standard_atomic_ref<u32_t const>(running_).load(std::memory_order_acquire);
    }

    /**
     *  @brief Starts the environment with @p config, unless it already runs.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k SPDK cannot take DPDK's environment in this process.
     *  - @c io_k The environment refused to start.
     */
    template <typename wait_type_>
    static status_t acquire(config_t const& config) noexcept {
        claim_<wait_type_>();
        status_t const acquired = running() ? status_t::success_k : start_(config);
        unclaim_();
        return acquired;
    }

    /**
     *  @brief A new holder of the controller @p transport_id names, attached on first use.
     *
     *  @par Statuses:
     *  - @c invalid_argument_k @p transport_id does not parse.
     *  - @c uninitialized_k The environment does not run.
     *  - @c io_k The controller did not connect, or failed and is still held.
     *  - @c bad_alloc_k The attachment could not be recorded.
     */
    template <typename wait_type_>
    static expected<connected_t*> connect(char const* transport_id) noexcept {
        spdk_nvme_transport_id transport {};
        if (!transport_id || spdk_nvme_transport_id_parse(&transport, transport_id) != 0)
            return {nullptr, status_t::invalid_argument_k};
        claim_<wait_type_>();
        expected<connected_t*> const connected = connect_(transport);
        unclaim_();
        return connected;
    }

    /** Adds a holder to @p connected, as an engine's queue pair on its controller does. */
    template <typename wait_type_>
    static void hold(connected_t& connected) noexcept {
        claim_<wait_type_>();
        ++connected.holders;
        unclaim_();
    }

    /** Drops a holder of @p connected, detaching its controller after the last one. */
    template <typename wait_type_>
    static void release(connected_t& connected) noexcept {
        claim_<wait_type_>();
        if (!--connected.holders) {
            for (std::size_t position = 0; position != connections_.size(); ++position)
                if (connections_[position] == &connected) {
                    connections_[position] = connections_[connections_.size() - 1];
                    connections_.pop_back();
                    break;
                }
            spdk_nvme_detach(connected.controller);
            delete &connected;
        }
        unclaim_();
    }

  private:
    template <typename wait_type_>
    static void claim_() noexcept {
        u32_t free = 0;
        while (!fu::standard_atomic_ref<u32_t>(claim_word_)
                    .compare_exchange_weak(free, 1, std::memory_order_acquire, std::memory_order_relaxed)) {
            wait_type_ {}(&claim_word_, free, 0u);
            free = 0;
        }
    }

    static void unclaim_() noexcept { fu::standard_atomic_ref<u32_t>(claim_word_).store(0, std::memory_order_release); }

    static status_t start_(config_t const& config) noexcept {
        if (!spdk_env_dpdk_external_init()) return status_t::config_mismatch_k;
        spdk_env_opts options {};
        options.opts_size = sizeof(options);
        spdk_env_opts_init(&options);
        options.name = "usearch";
        if (config.memory == linux_spdk_memory_t::plain_pages_k) {
            options.no_huge = true;
            options.no_pci = true;
            options.mem_size = static_cast<int>(config.plain_megabytes);
        }
        // The EAL pins the calling thread to its main core, and every thread spawned after would
        // inherit it.
        cpu_set_t inherited;
        int const kept_affinity = ::sched_getaffinity(0, sizeof(inherited), &inherited);
        int const initialized = spdk_env_init(&options);
        if (kept_affinity == 0) ::sched_setaffinity(0, sizeof(inherited), &inherited);
        if (initialized != 0) return status_t::io_k;
        fu::standard_atomic_ref<u32_t>(running_).store(1, std::memory_order_release);
        return status_t::success_k;
    }

    static expected<connected_t*> connect_(spdk_nvme_transport_id const& transport) noexcept {
        if (!running()) return {nullptr, status_t::uninitialized_k};
        for (connected_t* const cached : connections_)
            if (spdk_nvme_transport_id_compare(&cached->transport, &transport) == 0) {
                if (spdk_nvme_ctrlr_is_failed(cached->controller)) return {nullptr, status_t::io_k};
                ++cached->holders;
                return {cached, status_t::success_k};
            }
        spdk_nvme_ctrlr_opts options;
        spdk_nvme_ctrlr_get_default_ctrlr_opts(&options, sizeof(options));
        // Nothing polls the admin queue on a timer, so a fabric's keep-alive would drop the
        // controller.
        options.keep_alive_timeout_ms = 0;
        spdk_nvme_ctrlr* const controller = spdk_nvme_connect(&transport, &options, sizeof(options));
        if (!controller) return {nullptr, status_t::io_k};
        connected_t* const connected = new (std::nothrow) connected_t {transport, controller, 1};
        if (!connected || failed(connections_.push_back(connected)))
            return delete connected, spdk_nvme_detach(controller),
                   expected<connected_t*> {nullptr, status_t::bad_alloc_k};
        return {connected, status_t::success_k};
    }

    /** Held while the environment starts or a controller's holders change. */
    inline static u32_t claim_word_ = 0;

    /** One once the environment runs. */
    inline static u32_t running_ = 0;

    /** Every controller attached, found by transport ID. */
    inline static growing_buffer<connected_t*> connections_ {global_allocator};
};

/** One namespace on an SPDK controller, a holder of it until @c release, which must precede its
 *  death. */
struct spdk_medium_t {

    /** The controller the namespace sits on, null once released. */
    linux_spdk_process_t::connected_t* connection = nullptr;

    /** The namespace reads target. */
    spdk_nvme_ns* name_space = nullptr;

    /** Where the part's byte zero sits on the namespace. */
    u64_t byte_base = 0;

    /** The namespace's sector size. */
    u32_t block_bytes = 0;

    spdk_medium_t() noexcept = default;
    spdk_medium_t(spdk_medium_t&& other) noexcept
        : connection(std::exchange(other.connection, nullptr)), name_space(std::exchange(other.name_space, nullptr)),
          byte_base(std::exchange(other.byte_base, 0u)), block_bytes(std::exchange(other.block_bytes, 0u)) {}
    spdk_medium_t& operator=(spdk_medium_t&& other) noexcept {
        assert(!connection && "A held namespace is released before another takes its place");
        connection = std::exchange(other.connection, nullptr);
        name_space = std::exchange(other.name_space, nullptr);
        byte_base = std::exchange(other.byte_base, 0u);
        block_bytes = std::exchange(other.block_bytes, 0u);
        return *this;
    }
    spdk_medium_t(spdk_medium_t const&) = delete;
    spdk_medium_t& operator=(spdk_medium_t const&) = delete;
    ~spdk_medium_t() noexcept { assert(!connection && "A namespace is released before it dies"); }

    /** Gives back its hold on the controller, which detaches once no view or engine holds it. */
    template <typename wait_type_>
    void release() noexcept {
        if (connection) linux_spdk_process_t::release<wait_type_>(*std::exchange(connection, nullptr));
        name_space = nullptr;
    }

    /** Opens namespace @p namespace_id of the controller @p transport_id names, the first active
     *  one when zero, with the part starting @p byte_base bytes in, a multiple of its sector. */
    template <typename wait_type_>
    static expected<spdk_medium_t> open(char const* transport_id, u32_t namespace_id, u64_t byte_base) noexcept {
        spdk_medium_t medium;
        auto const [connection, status] = linux_spdk_process_t::connect<wait_type_>(transport_id);
        if (failed(status)) return {std::move(medium), status};
        medium.connection = connection;
        spdk_nvme_ctrlr* const controller = connection->controller;
        u32_t const identifier = namespace_id ? namespace_id : spdk_nvme_ctrlr_get_first_active_ns(controller);
        spdk_nvme_ns* const name_space = identifier ? spdk_nvme_ctrlr_get_ns(controller, identifier) : nullptr;
        u32_t const sector_bytes = name_space && spdk_nvme_ns_is_active(name_space)
                                       ? spdk_nvme_ns_get_sector_size(name_space)
                                       : 0;
        if (!sector_bytes || byte_base % sector_bytes)
            return medium.release<wait_type_>(), expected {std::move(medium), status_t::invalid_argument_k};
        medium.name_space = name_space;
        medium.byte_base = byte_base;
        medium.block_bytes = sector_bytes;
        return {std::move(medium), status_t::success_k};
    }

    /** Opens the namespace @p path names, an @c spdk:// address or a transport ID in any form
     *  @ref file_address_t lists, starting the process's environment over hugepages only for PCIe
     *  controllers. */
    template <typename wait_type_>
    static expected<spdk_medium_t> open(std::string_view path) noexcept {
        auto const [parsed, parse_status] = parse_file_address(path);
        if (failed(parse_status)) return {spdk_medium_t {}, parse_status};
        // The keys SPDK's parser does not know stay out of the transport ID it parses, and a bare
        // PCIe address of an spdk:// path gains the keys it lacks.
        std::string_view const pcie_keys = names_transport_id(parsed.location) ? "" : "trtype:PCIe traddr:";
        growing_buffer<char> transport_id {global_allocator};
        std::size_t const length = pcie_keys.size() + parsed.location.size();
        if (failed(transport_id.resize_uninitialized(length + 1))) return {spdk_medium_t {}, status_t::bad_alloc_k};
        parsed.location.copy(transport_id.data() + pcie_keys.copy(transport_id.data(), pcie_keys.size()), length);
        transport_id[length] = '\0';
        linux_spdk_process_t::config_t environment;
        if (pcie_keys.empty() && parsed.location.find("trtype:PCIe") == std::string_view::npos)
            environment.memory = linux_spdk_memory_t::plain_pages_k;
        if (status_t const acquired = linux_spdk_process_t::acquire<wait_type_>(environment); failed(acquired))
            return {spdk_medium_t {}, acquired};
        return open<wait_type_>(transport_id.data(), parsed.namespace_id, parsed.byte_base);
    }

    /** Fills @p bytes from @p offset bytes past the part's start, one sector-aligned read on a
     *  queue pair of its own waited out by polling, for the headers a view reads once. */
    status_t read_at(u64_t offset, bytes_span_t bytes) const noexcept {
        if (bytes.empty()) return status_t::success_k;
        spdk_nvme_qpair* const queue_pair = spdk_nvme_ctrlr_alloc_io_qpair(spdk_nvme_ns_get_ctrlr(name_space), nullptr,
                                                                           0);
        if (!queue_pair) return status_t::io_k;
        u64_t const begin = byte_base + offset, first = begin - begin % block_bytes;
        u64_t const bound = round_up_to_multiple<u64_t>(begin + bytes.size(), block_bytes);
        void* const staging = spdk_dma_malloc(bound - first, io_alignment_k, nullptr);
        // The command's callback writes the outcome; until then the read is still arriving.
        status_t read_status = status_t::uninitialized_k;
        if (!staging) read_status = status_t::bad_alloc_k;
        else if (spdk_nvme_ns_cmd_read(name_space, queue_pair, staging, first / block_bytes,
                                       static_cast<u32_t>((bound - first) / block_bytes),
                                       &spdk_medium_t::read_finished_, &read_status, 0) != 0)
            read_status = status_t::io_k;
        while (read_status == status_t::uninitialized_k)
            if (spdk_nvme_qpair_process_completions(queue_pair, 0) < 0) read_status = status_t::io_k;
        if (succeeded(read_status))
            std::memcpy(bytes.data(), static_cast<byte_t*>(staging) + (begin - first), bytes.size());
        spdk_nvme_ctrlr_free_io_qpair(queue_pair);
        if (staging) spdk_dma_free(staging);
        return read_status;
    }

  private:
    static void read_finished_(void* context, spdk_nvme_cpl const* completion) noexcept {
        *static_cast<status_t*>(context) = spdk_nvme_cpl_is_error(completion) ? status_t::io_k : status_t::success_k;
    }
};

/** The block every read on @p medium rounds its offset and length to. */
inline u32_t block_bytes_of(spdk_medium_t const& medium) noexcept { return medium.block_bytes; }

/** Where a part's byte zero sits on @p medium. */
inline u64_t byte_base_of(spdk_medium_t const& medium) noexcept { return medium.byte_base; }

/** Whether reads on @p medium skip the page cache, which a user-space driver never has. */
inline bool reads_directly(spdk_medium_t const&) noexcept { return true; }

/** The memory domain nearest the PCI function of the controller under @p medium; none off PCI. */
inline fu_memory_domain_id_t memory_domain_of(spdk_medium_t const& medium) noexcept {
    spdk_pci_device* const function = spdk_nvme_ctrlr_get_pci_device(spdk_nvme_ns_get_ctrlr(medium.name_space));
    if (!function) return no_memory_domain_k;
    spdk_pci_addr const address = spdk_pci_device_get_addr(function);
    char bus_id[sizeof("0000:00:00.0") + 4];
    std::snprintf(bus_id, sizeof(bus_id), "%04x:%02x:%02x.%x", static_cast<unsigned>(address.domain),
                  static_cast<unsigned>(address.bus), static_cast<unsigned>(address.dev),
                  static_cast<unsigned>(address.func));
    fu_memory_domain_id_t domain = no_memory_domain_k;
    return fu_memory_domain_id_of_pci(bus_id, &domain) == fu_success_k ? domain : no_memory_domain_k;
}

/** What the controller under @p medium granted this process, polled as SPDK always polls; its PCIe
 *  address stands for the disk, so a controller off PCIe reports none. */
inline io_device_queues_t device_queues_of(spdk_medium_t const& medium) noexcept {
    spdk_nvme_ctrlr* const controller = spdk_nvme_ns_get_ctrlr(medium.name_space);
    spdk_pci_device* const function = spdk_nvme_ctrlr_get_pci_device(controller);
    if (!function) return {};
    spdk_pci_addr const address = spdk_pci_device_get_addr(function);
    spdk_nvme_ctrlr_opts const& options = *spdk_nvme_ctrlr_get_opts(controller);
    io_device_queues_t queues;
    // Domain over bus over device and function, as Linux packs a PCI address.
    queues.device = address.domain << 16 | u32_t {address.bus} << 8 | u32_t {address.dev} << 3 | address.func;
    // Attach caps `io_queue_size` at MQES + 1, and a queue of N entries holds N - 1 commands.
    queues.tags = static_cast<u16_t>(options.io_queue_size - 1);
    // Attach lowers `num_io_queues` to what Set Features - Number of Queues allocated.
    queues.hardware_queues = static_cast<u16_t>(options.num_io_queues);
    queues.reaping = linux_uring_reaping_t::polled_k;
    queues.request_kibibytes = static_cast<u16_t>(
        min_of<u32_t>(spdk_nvme_ns_get_max_io_xfer_size(medium.name_space) / 1024, std::numeric_limits<u16_t>::max()));
    return queues;
}

#endif

/** The block every read on the medium @p medium holds rounds to. */
template <typename... media_types_>
u32_t block_bytes_of(std::variant<media_types_...> const& medium) noexcept {
    return std::visit([](auto const& alternative) noexcept { return block_bytes_of(alternative); }, medium);
}

/** Where a part's byte zero sits on the medium @p medium holds. */
template <typename... media_types_>
u64_t byte_base_of(std::variant<media_types_...> const& medium) noexcept {
    return std::visit([](auto const& alternative) noexcept { return byte_base_of(alternative); }, medium);
}

/** Whether reads on the medium @p medium holds skip the page cache. */
template <typename... media_types_>
bool reads_directly(std::variant<media_types_...> const& medium) noexcept {
    return std::visit([](auto const& alternative) noexcept { return reads_directly(alternative); }, medium);
}

/** The @p wanted_type_ medium @p medium is: itself, or the alternative a variant of media holds,
 *  null when it holds another transport's. */
template <typename wanted_type_>
wanted_type_ const* medium_as(wanted_type_ const& medium) noexcept {
    return &medium;
}

template <typename wanted_type_, typename... media_types_>
wanted_type_ const* medium_as(std::variant<media_types_...> const& medium) noexcept {
    return std::get_if<wanted_type_>(&medium);
}

/** Gives back the controller an SPDK namespace @p medium holds, which must precede its death;
 *  files and devices hold nothing shared. */
template <typename wait_type_, typename medium_type_>
void release_medium([[maybe_unused]] medium_type_& medium) noexcept {
#if USEARCH_WITH_LINUX_SPDK
    if constexpr (std::is_same_v<medium_type_, spdk_medium_t>) medium.template release<wait_type_>();
#endif
}

template <typename wait_type_, typename... media_types_>
void release_medium(std::variant<media_types_...>& medium) noexcept {
    std::visit([](auto& alternative) noexcept { release_medium<wait_type_>(alternative); }, medium);
}

/** What the block queue under the medium @p medium holds admits. */
template <typename... media_types_>
io_device_queues_t device_queues_of(std::variant<media_types_...> const& medium) noexcept {
    return std::visit([](auto const& alternative) noexcept { return device_queues_of(alternative); }, medium);
}

/** The reads each of @p threads engines keeps in flight over @p media without waiting on a tag:
 *  the sum over distinct disks of their tags divided by the threads sharing each hardware queue,
 *  as blk-mq spreads CPUs evenly over a disk's queues; zero when no medium has a block queue. */
template <typename media_type_>
std::size_t nvme_queue_depth(std::span<media_type_ const> media, std::size_t threads) noexcept {
    assert(threads && "Depth is shared among at least one thread");
    std::size_t depth = 0;
    for (std::size_t position = 0; position != media.size(); ++position) {
        io_device_queues_t const queues = device_queues_of(media[position]);
        std::size_t earlier = 0;
        while (earlier != position && device_queues_of(media[earlier]).device != queues.device) ++earlier;
        if (!queues.device || earlier != position) continue;
        depth += queues.tags / divide_round_up<std::size_t>(threads, queues.hardware_queues);
    }
    return depth;
}

/** Polled reaping when every medium of @p media polls, interrupts otherwise. */
template <typename media_type_>
linux_uring_reaping_t nvme_reaping(std::span<media_type_ const> media) noexcept {
    for (media_type_ const& medium : media)
        if (device_queues_of(medium).reaping != linux_uring_reaping_t::polled_k)
            return linux_uring_reaping_t::interrupted_k;
    return media.empty() ? linux_uring_reaping_t::interrupted_k : linux_uring_reaping_t::polled_k;
}

#pragma endregion Media

#pragma region Engine Contract

/** How an engine opens; each engine reads the fields its transport knows. */
struct io_engine_config_t {

    /** Reads outstanding at most between @c enqueue_read and @c poll_completions; an executor
     *  spawned with zero lets each call derive it from the block queues of its media. */
    std::size_t queue_depth = 0;

    /** The transport an engine set opens. */
    io_transport_t transport = io_transport_t::automatic_k;

    /** Whether io_uring reads arrive in registered buffers; passthrough honors @c fixed_k from
     *  Linux 6.14. */
    linux_uring_buffer_registration_t buffer_registration = linux_uring_buffer_registration_t::fixed_k;

    /** Whether io_uring learns of completions by interrupt or by polling the device's queues; an
     *  executor's calls derive it from their media through @c nvme_reaping. */
    linux_uring_reaping_t reaping = linux_uring_reaping_t::interrupted_k;
};

/** What an engine driving @p transport answers @p config before setting anything up:
 *  @c invalid_argument_k without a queue depth, and @c unsupported_k when the configuration names
 *  another transport. */
constexpr status_t io_engine_vet_(io_engine_config_t const& config, io_transport_t transport) noexcept {
    if (!config.queue_depth) return status_t::invalid_argument_k;
    if (config.transport != io_transport_t::automatic_k && config.transport != transport)
        return status_t::unsupported_k;
    return status_t::success_k;
}

/** What an engine holding @p media_count attached media answers @p request before encoding it:
 *  @c invalid_argument_k for a medium it never attached, or for a length past what a completion's
 *  result counts. */
inline status_t io_request_vet_(io_request_t const& request, std::size_t media_count) noexcept {
    assert(request.length <= io_max_length_k && "A completion's result must count every byte its read moves");
    if (request.medium >= media_count || request.length > io_max_length_k) return status_t::invalid_argument_k;
    return status_t::success_k;
}

/**
 *  @brief The contract every I/O engine meets, so a walk drives any of them unchanged.
 *
 *  @c reserve grows the staging buffers, re-registering them only on growth; @c attach makes
 *  request medium IDs index the given media until @c detach; @c enqueue_read only encodes;
 *  @c submit hands every encoded read to the device and, under @c blocking_k, holds for a
 *  completion; @c poll_completions moves finished reads out; @c buffer maps an index to the memory
 *  a read arrived in.
 *
 *  Every read @c enqueue_read accepts yields exactly one completion, naming its buffer and its
 *  context, and that buffer serves no other read until the completion is polled. A failed
 *  @c submit is the one exception: it waits out every read the kernel holds and drops the rest, so
 *  it returns with nothing in flight and nothing left to poll, and the call that saw it fails.
 *  @c reserve and @c attach run with nothing outstanding, on the thread that drives the engine.
 */
template <typename engine_type_>
concept io_engine = requires(engine_type_& engine, io_request_t request, std::span<io_completion_t> completions,
                             io_wait_t wait, io_buffer_id_t buffer,
                             std::span<typename engine_type_::medium_t const> media, std::size_t buffers,
                             std::size_t buffer_bytes) {
    { engine.reserve(buffers, buffer_bytes) } noexcept -> std::same_as<status_t>;
    { engine.attach(media) } noexcept -> std::same_as<status_t>;
    { engine.detach() } noexcept -> std::same_as<void>;
    { engine.enqueue_read(request) } noexcept -> std::same_as<status_t>;
    { engine.submit(wait) } noexcept -> std::same_as<status_t>;
    { engine.poll_completions(completions) } noexcept -> std::same_as<std::size_t>;
    { engine.buffer(buffer) } noexcept -> std::same_as<std::byte*>;
};

/** How a staging region is cut: its buffers and the bytes of each. */
struct io_buffers_cut_t {
    std::size_t buffers = 0;
    std::size_t buffer_bytes = 0;
};

/** The staging buffers an engine owns: one region of whole pages, placed through its allocator,
 *  which aligns it to a page, cut per call into equal strides on @c io_alignment_k, and regrown
 *  only when a call needs more bytes than it holds. */
template <typename allocator_type_>
class io_buffers {
  public:
    explicit io_buffers(allocator_type_ const& allocator) noexcept : bytes_(allocator) {}
    io_buffers(io_buffers&& other) noexcept
        : bytes_(std::move(other.bytes_)), base_(std::exchange(other.base_, nullptr)),
          capacity_(std::exchange(other.capacity_, 0u)), count_(std::exchange(other.count_, 0u)),
          stride_(std::exchange(other.stride_, 0u)) {}
    io_buffers& operator=(io_buffers&& other) noexcept {
        bytes_ = std::move(other.bytes_);
        std::swap(base_, other.base_);
        std::swap(capacity_, other.capacity_);
        std::swap(count_, other.count_);
        std::swap(stride_, other.stride_);
        return *this;
    }

    /** Whether @p count buffers of @p bytes each fit the region already held. */
    bool covers(std::size_t count, std::size_t bytes) const noexcept { return count <= capacity_ / stride_of_(bytes); }

    /**
     *  @brief Cuts the region into @p count buffers of @p bytes each, regrowing it only when they
     *      do not fit, which moves every buffer, so nothing may be in flight; empties on failure.
     *
     *  @par Statuses:
     *  - @c overflow_k The region's byte count overflows.
     *  - @c config_mismatch_k The allocator placed the region off an @c io_alignment_k boundary.
     */
    status_t reserve(std::size_t count, std::size_t bytes) noexcept {
        std::size_t const stride = stride_of_(bytes);
        if (covers(count, bytes)) return count_ = count, stride_ = stride, status_t::success_k;
        reset();
        std::size_t const page_bytes = fu::ram_page_size();
        if (count > (~std::size_t {0} - page_bytes) / stride) return status_t::overflow_k;
        // Whole pages, which the allocator starts on a page, so the region is taken as placed.
        std::size_t const region_bytes = round_up_to_multiple<std::size_t>(max_of<std::size_t>(count * stride, 1),
                                                                           page_bytes);
        if (status_t const grown = bytes_.resize_uninitialized(region_bytes); failed(grown)) return grown;
        if (reinterpret_cast<std::uintptr_t>(bytes_.data()) % io_alignment_k)
            return reset(), status_t::config_mismatch_k;
        base_ = bytes_.data();
        capacity_ = region_bytes, count_ = count, stride_ = stride;
        return status_t::success_k;
    }

    /** Releases every buffer. */
    void reset() noexcept {
        bytes_.reset();
        base_ = nullptr, capacity_ = 0, count_ = 0, stride_ = 0;
    }

    byte_t* buffer(io_buffer_id_t buffer) const noexcept { return base_ + static_cast<std::size_t>(buffer) * stride_; }
    std::size_t count() const noexcept { return count_; }
    std::size_t stride() const noexcept { return stride_; }

    /** The whole region, which the fixed-buffer paths register once. */
    byte_t* base() const noexcept { return base_; }
    std::size_t capacity() const noexcept { return capacity_; }

  private:
    static std::size_t stride_of_(std::size_t bytes) noexcept {
        return round_up_to_multiple<std::size_t>(max_of<std::size_t>(bytes, 1), io_alignment_k);
    }

    growing_buffer<byte_t, allocator_type_> bytes_;
    byte_t* base_ = nullptr;
    std::size_t capacity_ = 0;
    std::size_t count_ = 0;
    std::size_t stride_ = 0;
};

/** Copies the descriptors of @p media into @p descriptors by medium ID; a medium of another
 *  transport answers @c config_mismatch_k. */
template <typename allocator_type_, typename media_type_>
status_t copy_descriptors_(growing_buffer<native_file_t, allocator_type_>& descriptors,
                           std::span<media_type_ const> media) noexcept {
    descriptors.clear();
    if (status_t const reserved = descriptors.reserve(media.size()); failed(reserved)) return reserved;
    for (media_type_ const& medium : media) {
        file_medium_t const* const file = medium_as<file_medium_t>(medium);
        if (!file) return descriptors.clear(), status_t::config_mismatch_k;
        descriptors.push_back_reserved(file->descriptor);
    }
    return status_t::success_k;
}

/** Moves as many queued completions as @p completions holds off the back of @p queue, answering how
 *  many. */
template <typename allocator_type_>
std::size_t poll_queued_completions_(growing_buffer<io_completion_t, allocator_type_>& queue,
                                     std::span<io_completion_t> completions) noexcept {
    std::size_t const polled = min_of(queue.size(), completions.size());
    if (!polled) return 0;
    std::memcpy(completions.data(), queue.end() - polled, polled * sizeof(io_completion_t));
    queue.resize_reserved(queue.size() - polled);
    return polled;
}

#pragma endregion Engine Contract

#pragma region Blocking and AIO Engines

inline status_t file_medium_t::read_at(u64_t offset, bytes_span_t bytes) const noexcept {
    if (bytes.empty()) return status_t::success_k;
    u64_t const begin = byte_base + offset, first = begin - begin % block_bytes;
    u64_t const bound = round_up_to_multiple<u64_t>(begin + bytes.size(), block_bytes);
    if (bound - first > io_max_length_k) return status_t::overflow_k;
    io_buffers<forkunion_allocator_t> staging {forkunion_allocator_t {}};
    if (status_t const reserved = staging.reserve(1, bound - first); failed(reserved)) return reserved;
    u32_t const length = static_cast<u32_t>(bound - first);
    i64_t moved = read_native_file(descriptor, native_file_event_t {}, {staging.buffer(0), length}, first);
    // A generic char node has no read(2); its namespace serves the same blocks to NVMe Reads.
    if (moved == -EINVAL) moved = nvme_namespace_read_(descriptor, {staging.buffer(0), length}, first, block_bytes);
    // The file may end inside its last block, so only the bytes asked for must arrive.
    if (moved < 0 || static_cast<u64_t>(moved) < begin + bytes.size() - first) return status_t::io_k;
    std::memcpy(bytes.data(), staging.buffer(0) + (begin - first), bytes.size());
    return status_t::success_k;
}

/**
 *  @brief Portable blocking engine: @c submit services every enqueued read with one @c pread or
 *      @c ReadFile apiece on the calling thread.
 *
 *  The reference implementation of the concept, the debugging harness, and the macOS production
 *  path, where a pool of threads, each driving its own engine, is the realistic ceiling anyway.
 *
 *  macOS specifics: there is no async-file syscall surface, since POSIX AIO is capped at a queue
 *  depth of ~16 by @c kern.aiomax -class sysctls and kqueue cannot watch regular files, so threads
 *  x @c pread with @c F_NOCACHE is the platform idiom. Note @c F_NOCACHE is a hint, not
 *  @c O_DIRECT: unaligned I/O silently falls back to the buffered path, and resident pages are
 *  still served from cache.
 */
template <typename wait_type_, typename allocator_type_ = forkunion_allocator_t>
class blocking_files_engine {
  public:
    using wait_t = wait_type_;
    using allocator_t = allocator_type_;
    using medium_t = file_medium_t;
    using config_t = io_engine_config_t;

    static constexpr io_transport_t transport() noexcept { return io_transport_t::blocking_k; }

    explicit blocking_files_engine(allocator_t const& allocator = {}) noexcept
        : buffers_(allocator), descriptors_(allocator), pending_(allocator), completions_(allocator) {}
    blocking_files_engine(blocking_files_engine&&) noexcept = default;
    blocking_files_engine& operator=(blocking_files_engine&&) noexcept = default;

    /**
     *  Sizes the request and completion queues for @p config's depth.
     *
     *  @par Statuses:
     *  - @c invalid_argument_k A zero queue depth.
     *  - @c unsupported_k @p config names another transport.
     */
    status_t open(config_t const& config) noexcept {
        if (status_t const vetted = io_engine_vet_(config, transport()); failed(vetted)) return vetted;
        if (failed(pending_.reserve_exactly(config.queue_depth)) ||
            failed(completions_.reserve_exactly(config.queue_depth)))
            return status_t::bad_alloc_k;
        queue_depth_ = config.queue_depth;
        return status_t::success_k;
    }

    /**
     *  Cuts the staging region into @p buffers buffers of @p buffer_bytes each.
     *
     *  @par Statuses:
     *  - @c overflow_k The region's byte count overflows.
     */
    status_t reserve(std::size_t buffers, std::size_t buffer_bytes) noexcept {
        return buffers_.reserve(buffers, buffer_bytes);
    }

    /**
     *  Takes the descriptors of @p media, which request medium IDs index until @c detach.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k A medium that is not a file.
     */
    template <typename media_type_>
    status_t attach(std::span<media_type_ const> media) noexcept {
        return copy_descriptors_(descriptors_, media);
    }

    void detach() noexcept { descriptors_.clear(); }

    /**
     *  Queues @p request for the next @c submit.
     *
     *  @par Statuses:
     *  - @c overflow_k The queue already holds its depth of requests and completions.
     *  - @c invalid_argument_k A medium never attached, or a length past what a completion counts.
     */
    status_t enqueue_read(io_request_t request) noexcept {
        if (pending_.size() + completions_.size() == queue_depth_) return status_t::overflow_k;
        if (status_t const vetted = io_request_vet_(request, descriptors_.size()); failed(vetted)) return vetted;
        pending_.push_back_reserved(request);
        return status_t::success_k;
    }

    /** Reads everything enqueued before returning, so waiting is implicit. */
    status_t submit(io_wait_t) noexcept {
        for (io_request_t const& request : pending_) {
            i32_t const result = static_cast<i32_t>(read_native_file(descriptors_[request.medium], event_,
                                                                     {buffers_.buffer(request.buffer), request.length},
                                                                     request.offset));
            completions_.push_back_reserved(io_completion_t {request.buffer, result, request.context});
        }
        pending_.clear();
        return status_t::success_k;
    }

    std::size_t poll_completions(std::span<io_completion_t> completions) noexcept {
        return poll_queued_completions_(completions_, completions);
    }

    std::byte* buffer(io_buffer_id_t buffer) noexcept { return buffers_.buffer(buffer); }

  private:

    /** The staging buffers reads arrive in. */
    io_buffers<allocator_t> buffers_;

    /** The attached media's descriptors, indexed by medium ID. */
    growing_buffer<native_file_t, allocator_t> descriptors_;

    /** Reads enqueued since the last @c submit. */
    growing_buffer<io_request_t, allocator_t> pending_;

    /** Reads finished and not yet polled. */
    growing_buffer<io_completion_t, allocator_t> completions_;

    /** The event this engine's thread waits its reads out on, empty off Windows. */
    [[no_unique_address]] native_file_event_t event_;

    /** Reads outstanding at most; zero until @c open. */
    std::size_t queue_depth_ = 0;
};

#if USEARCH_WITH_LINUX_AIO

/** One AIO context, destroyed with its owner; @c io_destroy waits out every read still in
 *  flight. */
struct linux_aio_context_t {

    /** The kernel's handle, zero while none is set up. */
    aio_context_t context = 0;

    linux_aio_context_t() noexcept = default;
    linux_aio_context_t(linux_aio_context_t&& other) noexcept : context(std::exchange(other.context, 0u)) {}
    linux_aio_context_t& operator=(linux_aio_context_t&& other) noexcept {
        std::swap(context, other.context);
        return *this;
    }
    linux_aio_context_t(linux_aio_context_t const&) = delete;
    linux_aio_context_t& operator=(linux_aio_context_t const&) = delete;
    ~linux_aio_context_t() noexcept { destroy(); }

    /** Sets up a context admitting @p events reads in flight. */
    status_t setup(std::size_t events) noexcept {
        destroy();
        long const created = syscall(SYS_io_setup, static_cast<unsigned>(events), &context);
        return created == 0 ? status_t::success_k : status_t::io_k;
    }

    /** Waits out every read in flight and drops the context. */
    void destroy() noexcept {
        if (context) syscall(SYS_io_destroy, context);
        context = 0;
    }
};

/**
 *  @brief Linux AIO engine over raw @c io_submit / @c io_getevents syscalls, with @c O_DIRECT
 *      media.
 *
 *  Roughly two thirds of io_uring's per-core throughput, but its syscalls predate every seccomp
 *  profile in existence: the engine for container platforms and hardened fleets where io_uring is
 *  administratively gated. Default Docker/containerd seccomp profiles block the io_uring family,
 *  gVisor rejects it, and the 6.6+ @c kernel.io_uring_disabled sysctl can turn it off fleet-wide.
 *
 *  The @c iocb fields a read consumes: @c aio_fildes, @c aio_lio_opcode, @c aio_buf, @c aio_nbytes,
 *  @c aio_offset & @c aio_data; the @c io_event returns @c data & @c res. True asynchrony holds
 *  only on the @c O_DIRECT path, since buffered submissions can block inside @c io_submit; that
 *  costs this design nothing, as every read is already block-aligned.
 */
template <typename wait_type_, typename allocator_type_ = forkunion_allocator_t>
class linux_aio_files_engine {
  public:
    using wait_t = wait_type_;
    using allocator_t = allocator_type_;
    using medium_t = file_medium_t;
    using config_t = io_engine_config_t;

    static constexpr io_transport_t transport() noexcept { return io_transport_t::linux_aio_k; }

    explicit linux_aio_files_engine(allocator_t const& allocator = {}) noexcept
        : buffers_(allocator), request_contexts_(allocator), descriptors_(allocator), control_blocks_(allocator),
          control_pointers_(allocator), events_(allocator), completions_(allocator) {}
    linux_aio_files_engine(linux_aio_files_engine&&) noexcept = default;
    linux_aio_files_engine& operator=(linux_aio_files_engine&&) noexcept = default;

    /** Sets up the context and sizes every queue for @p config's depth. */
    status_t open(config_t const& config) noexcept {
        if (status_t const vetted = io_engine_vet_(config, transport()); failed(vetted)) return vetted;
        if (failed(control_blocks_.reserve_exactly(config.queue_depth)) ||
            failed(control_pointers_.reserve_exactly(config.queue_depth)) ||
            failed(events_.resize(config.queue_depth)) || failed(completions_.reserve_exactly(config.queue_depth)))
            return status_t::bad_alloc_k;
        if (status_t const created = context_.setup(config.queue_depth); failed(created)) return created;
        queue_depth_ = config.queue_depth;
        return status_t::success_k;
    }

    status_t reserve(std::size_t buffers, std::size_t buffer_bytes) noexcept {
        if (status_t const grown = buffers_.reserve(buffers, buffer_bytes); failed(grown)) return grown;
        return request_contexts_.size() >= buffers ? status_t::success_k : request_contexts_.resize(buffers);
    }

    template <typename media_type_>
    status_t attach(std::span<media_type_ const> media) noexcept {
        return copy_descriptors_(descriptors_, media);
    }

    void detach() noexcept { descriptors_.clear(); }

    /** Encodes the control block in place, so @c submit only submits. */
    status_t enqueue_read(io_request_t request) noexcept {
        if (!context_.context) return status_t::io_k;
        if (control_blocks_.size() + in_flight_ + completions_.size() == queue_depth_) return status_t::overflow_k;
        if (status_t const vetted = io_request_vet_(request, descriptors_.size()); failed(vetted)) return vetted;
        struct iocb control {};
        control.aio_data = request.buffer;
        control.aio_lio_opcode = IOCB_CMD_PREAD;
        control.aio_fildes = static_cast<u32_t>(descriptors_[request.medium]);
        control.aio_buf = reinterpret_cast<std::uintptr_t>(buffers_.buffer(request.buffer));
        control.aio_nbytes = request.length;
        control.aio_offset = static_cast<i64_t>(request.offset);
        control_blocks_.push_back_reserved(control);
        control_pointers_.push_back_reserved(&control_blocks_.back());
        request_contexts_[request.buffer] = request.context;
        return status_t::success_k;
    }

    status_t submit(io_wait_t wait) noexcept {
        submit_staged_();
        if (!in_flight_) return status_t::success_k;
        long const minimum = wait == io_wait_t::blocking_k && completions_.empty() ? 1 : 0;
        // AIO posts no completion word to watch, so one `io_getevents` both polls and parks.
        long const reaped = syscall(SYS_io_getevents, context_.context, minimum, static_cast<long>(in_flight_),
                                    events_.data(), nullptr);
        if (reaped < 0 && errno == EINTR) return status_t::success_k;
        if (reaped < 0) return abandon_();
        for (long event = 0; event != reaped; ++event)
            complete_(static_cast<io_buffer_id_t>(events_[event].data), static_cast<i32_t>(events_[event].res));
        in_flight_ -= static_cast<std::size_t>(reaped);
        return status_t::success_k;
    }

    std::size_t poll_completions(std::span<io_completion_t> completions) noexcept {
        return poll_queued_completions_(completions_, completions);
    }

    std::byte* buffer(io_buffer_id_t buffer) noexcept { return buffers_.buffer(buffer); }

  private:

    /** Queues the completion of the read @p buffer served, with @p result and its context. */
    void complete_(io_buffer_id_t buffer, i32_t result) noexcept {
        completions_.push_back_reserved(io_completion_t {buffer, result, request_contexts_[buffer]});
    }

    /** Hands every staged control block to the kernel; a block it refuses fails alone, so its
     *  neighbors still go. */
    void submit_staged_() noexcept {
        std::size_t const staged = control_pointers_.size();
        for (std::size_t submitted = 0; submitted != staged;) {
            long const accepted = syscall(SYS_io_submit, context_.context, static_cast<long>(staged - submitted),
                                          control_pointers_.data() + submitted);
            if (accepted > 0) {
                submitted += static_cast<std::size_t>(accepted);
                in_flight_ += static_cast<std::size_t>(accepted);
                continue;
            }
            i32_t const error = accepted < 0 ? errno : EIO;
            complete_(static_cast<io_buffer_id_t>(control_blocks_[submitted].aio_data), -error);
            ++submitted;
        }
        control_blocks_.clear();
        control_pointers_.clear();
    }

    /** Waits out every read the kernel holds by dropping the context, empties the engine, and sets
     *  up a fresh context for the next call. */
    status_t abandon_() noexcept {
        context_.destroy();
        completions_.clear();
        in_flight_ = 0;
        // A context that cannot come back fails every later read rather than reporting the engine
        // full.
        [[maybe_unused]] status_t const reopened = context_.setup(queue_depth_);
        return status_t::io_k;
    }

    /** The staging buffers reads arrive in. */
    io_buffers<allocator_t> buffers_;

    /** The context of the read each buffer serves, indexed by buffer. */
    growing_buffer<u64_t, allocator_t> request_contexts_;

    /** The attached media's descriptors, indexed by medium ID. */
    growing_buffer<native_file_t, allocator_t> descriptors_;

    /** Control blocks encoded since the last @c submit; the kernel copies them on @c io_submit. */
    growing_buffer<struct iocb, allocator_t> control_blocks_;

    /** The pointer array @c io_submit takes, one per staged control block. */
    growing_buffer<struct iocb*, allocator_t> control_pointers_;

    /** The completion array of @c io_getevents, one slot per read the context admits. */
    growing_buffer<struct io_event, allocator_t> events_;

    /** Reads finished and not yet polled. */
    growing_buffer<io_completion_t, allocator_t> completions_;

    /** The context, declared after the buffers so it dies first and waits out reads still arriving
     *  in them. */
    linux_aio_context_t context_;

    /** Reads outstanding at most, staged, held by the kernel or finished and unpolled; zero until
     *  @c open. */
    std::size_t queue_depth_ = 0;

    /** Reads the kernel holds. */
    std::size_t in_flight_ = 0;
};

#endif

#pragma endregion Blocking and AIO Engines

#pragma region Ring Engines

#if USEARCH_WITH_LINUX_URING

/** SQE geometry of one ring: 64-byte entries for block-layer reads, or the
 *  128-byte form whose inline command area carries NVMe passthrough. */
enum class linux_uring_form_t : u32_t {
    compact_k,
    nvme_k,
};

/** When CQEs become visible: @c immediate_k rings post them asynchronously, so a waiter can monitor
 *  the CQ tail from user space, while @c deferred_k rings run completion work only inside
 *  @c io_uring_enter, the single-issuer deferred-task-run discipline, making tail spinning
 *  pointless. */
enum class linux_uring_posting_t : u32_t {
    immediate_k,
    deferred_k,
};

/** One @c mmap -ed ring region; unmaps itself, so teardown is member order. */
struct linux_uring_mapping_t {

    /** The mapped address, null while nothing is mapped. */
    void* base = nullptr;

    /** Bytes mapped. */
    std::size_t bytes = 0;

    linux_uring_mapping_t() noexcept = default;
    linux_uring_mapping_t(linux_uring_mapping_t&& other) noexcept
        : base(std::exchange(other.base, nullptr)), bytes(std::exchange(other.bytes, 0u)) {}
    linux_uring_mapping_t& operator=(linux_uring_mapping_t&& other) noexcept {
        std::swap(base, other.base);
        std::swap(bytes, other.bytes);
        return *this;
    }
    linux_uring_mapping_t(linux_uring_mapping_t const&) = delete;
    linux_uring_mapping_t& operator=(linux_uring_mapping_t const&) = delete;
    ~linux_uring_mapping_t() noexcept { reset(); }

    status_t map(int ring_descriptor, std::size_t map_bytes, u64_t region) noexcept {
        base = ::mmap(nullptr, map_bytes, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, ring_descriptor,
                      static_cast<off_t>(region));
        if (base == MAP_FAILED) return base = nullptr, status_t::io_k;
        bytes = map_bytes;
        return status_t::success_k;
    }
    void reset() noexcept {
        if (base) ::munmap(base, bytes);
        base = nullptr, bytes = 0;
    }
    u32_t* word(u32_t offset) const noexcept { return reinterpret_cast<u32_t*>(static_cast<byte_t*>(base) + offset); }
};

/** Everything the submission side of one ring needs: the kernel-shared head &
 *  tail words, the index-order array, the wrap mask, and the SQE array. */
struct linux_uring_submissions_t {
    u32_t* head_word = nullptr;
    u32_t* tail_word = nullptr;
    u32_t* order = nullptr;
    u32_t mask = 0;
    byte_t* entries = nullptr;
    std::size_t stride = 0;
};

/** The completion-side mirror: head & tail words, wrap mask, and CQE array. */
struct linux_uring_completions_t {
    u32_t* head_word = nullptr;
    u32_t* tail_word = nullptr;
    u32_t mask = 0;
    byte_t* entries = nullptr;
    std::size_t stride = 0;
};

/**
 *  @brief Shared io_uring plumbing of the Linux ring engines.
 *
 *  Setup with graceful flag degradation, the ring mappings, submission-queue slot claims, resource
 *  registration, and completion-queue draining, all parameterized by SQE and CQE strides at
 *  runtime, since NVMe passthrough doubles both.
 *
 *  Setup prefers @c IORING_SETUP_SINGLE_ISSUER with @c IORING_SETUP_DEFER_TASKRUN, the 6.1+ mode
 *  where completion work runs batched exactly when the owning thread asks, and falls back to a
 *  plain ring on older kernels. Two consequences matter to callers: under @c DEFER_TASKRUN
 *  completions only materialize inside @c io_uring_enter, so spinning on the CQ tail from user
 *  space is futile; and under @c SINGLE_ISSUER every enter and every registration comes from the
 *  thread that created the ring.
 */
class linux_uring_ring_t {
  public:

    /** The registered-ring index of a ring whose descriptor the kernel declined to register. */
    inline static constexpr u32_t unregistered_ring_k = ~u32_t {0};

    linux_uring_ring_t() noexcept = default;
    linux_uring_ring_t(linux_uring_ring_t&& other) noexcept { swap(other); }
    linux_uring_ring_t& operator=(linux_uring_ring_t&& other) noexcept {
        swap(other);
        return *this;
    }
    linux_uring_ring_t(linux_uring_ring_t const&) = delete;
    linux_uring_ring_t& operator=(linux_uring_ring_t const&) = delete;
    ~linux_uring_ring_t() noexcept {
        if (ring_descriptor_ < 0) return;
        // A closed ring unpins its buffers in deferred work, holding memlock that later rings need.
        unregister_buffers();
        ::close(ring_descriptor_);
    }

    void swap(linux_uring_ring_t& other) noexcept {
        std::swap(ring_descriptor_, other.ring_descriptor_);
        std::swap(reaping_, other.reaping_);
        std::swap(posting_, other.posting_);
        std::swap(submission_mapping_, other.submission_mapping_);
        std::swap(completion_mapping_, other.completion_mapping_);
        std::swap(entries_mapping_, other.entries_mapping_);
        std::swap(submissions_, other.submissions_);
        std::swap(completions_, other.completions_);
        std::swap(pending_tail_, other.pending_tail_);
        std::swap(registered_ring_index_, other.registered_ring_index_);
        std::swap(registered_regions_, other.registered_regions_);
        std::swap(queue_depth_, other.queue_depth_);
        std::swap(outstanding_, other.outstanding_);
    }

    /**
     *  @brief Creates the ring & maps its regions.
     *
     *  @param[in] queue_depth Submission-queue entries, and the reads the ring holds at most;
     *      completions sized 2x by the kernel.
     *  @param[in] form SQE & CQE strides: compact, or doubled for NVMe passthrough.
     *  @param[in] reaping Interrupt-driven completions, or device-queue polling, which a refused
     *      IOPOLL ring trades for interrupts.
     *
     *  SQE128 and CQE32 need Linux 5.19, SINGLE_ISSUER 6.0 and DEFER_TASKRUN 6.1.
     */
    status_t setup(std::size_t queue_depth, linux_uring_form_t form, linux_uring_reaping_t reaping) noexcept {
        long const setup_syscall = 425 + linux_syscall_offset_k; // __NR_io_uring_setup

        u32_t const iopoll = 1u << 0;                   // IORING_SETUP_IOPOLL
        u32_t const sqe128 = 1u << 10;                  // IORING_SETUP_SQE128
        u32_t const cqe32 = 1u << 11;                   // IORING_SETUP_CQE32
        u32_t const single_issuer = 1u << 12;           // IORING_SETUP_SINGLE_ISSUER
        u32_t const defer_taskrun = 1u << 13;           // IORING_SETUP_DEFER_TASKRUN
        u32_t const single_mmap = 1u << 0;              // IORING_FEAT_SINGLE_MMAP
        u64_t const submission_ring = 0ull;             // IORING_OFF_SQ_RING
        u64_t const completion_ring = 0x8000000ull;     // IORING_OFF_CQ_RING
        u64_t const submission_entries = 0x10000000ull; // IORING_OFF_SQES
        linux_uring_params_t params;
        u32_t const interrupted_flags = form == linux_uring_form_t::nvme_k ? sqe128 | cqe32 : 0u;
        u32_t const shape_flags = interrupted_flags | (reaping == linux_uring_reaping_t::polled_k ? iopoll : 0u);
        u32_t const deferred_flags = single_issuer | defer_taskrun;
        // A refused IOPOLL ring falls to interrupts, as a refused deferred task-run does to plain.
        for (u32_t const flags :
             {shape_flags | deferred_flags, shape_flags, interrupted_flags | deferred_flags, interrupted_flags}) {
            params = linux_uring_params_t {};
            params.setup_flags = flags;
            long const ring_descriptor = syscall(setup_syscall, static_cast<unsigned>(queue_depth), &params);
            if (ring_descriptor >= 0) {
                ring_descriptor_ = static_cast<int>(ring_descriptor);
                break;
            }
            // Kernels before 5.19 have no 128-byte entries, so they leave passthrough unsupported.
            if (flags == interrupted_flags)
                return form == linux_uring_form_t::nvme_k ? status_t::unsupported_k : status_t::io_k;
        }
        reaping_ = params.setup_flags & iopoll ? linux_uring_reaping_t::polled_k : linux_uring_reaping_t::interrupted_k;
        posting_ = params.setup_flags & defer_taskrun ? linux_uring_posting_t::deferred_k
                                                      : linux_uring_posting_t::immediate_k;
        submissions_.stride = form == linux_uring_form_t::nvme_k ? sizeof(linux_uring_sqe128_t)
                                                                 : sizeof(linux_uring_sqe_t);
        completions_.stride = form == linux_uring_form_t::nvme_k ? 2 * sizeof(linux_uring_cqe_t)
                                                                 : sizeof(linux_uring_cqe_t);

        // Map the two rings & the SQE array. Modern kernels serve SQ & CQ from one mapping.
        std::size_t submission_bytes = params.sq_off.array_offset + params.sq_entries * sizeof(u32_t);
        std::size_t const completion_bytes = params.cq_off.cqes_offset + params.cq_entries * completions_.stride;
        bool const single_mapping = params.features & single_mmap;
        if (single_mapping) submission_bytes = max_of(submission_bytes, completion_bytes);
        status_t mapped = submission_mapping_.map(ring_descriptor_, submission_bytes, submission_ring);
        if (failed(mapped)) return mapped;
        linux_uring_mapping_t const& completion_mapping = single_mapping ? submission_mapping_ : completion_mapping_;
        if (!single_mapping) {
            mapped = completion_mapping_.map(ring_descriptor_, completion_bytes, completion_ring);
            if (failed(mapped)) return mapped;
        }
        mapped = entries_mapping_.map(ring_descriptor_, params.sq_entries * submissions_.stride, submission_entries);
        if (failed(mapped)) return mapped;

        submissions_.head_word = submission_mapping_.word(params.sq_off.head_offset);
        submissions_.tail_word = submission_mapping_.word(params.sq_off.tail_offset);
        submissions_.order = submission_mapping_.word(params.sq_off.array_offset);
        submissions_.mask = *submission_mapping_.word(params.sq_off.ring_mask_offset);
        submissions_.entries = static_cast<byte_t*>(entries_mapping_.base);
        completions_.head_word = completion_mapping.word(params.cq_off.head_offset);
        completions_.tail_word = completion_mapping.word(params.cq_off.tail_offset);
        completions_.mask = *completion_mapping.word(params.cq_off.ring_mask_offset);
        completions_.entries = static_cast<byte_t*>(completion_mapping.base) + params.cq_off.cqes_offset;
        pending_tail_ = *submissions_.tail_word;
        queue_depth_ = queue_depth;
        register_ring_descriptor_();
        return status_t::success_k;
    }

    /** Pins @p regions into the ring's registered-buffer table, so @c READ_FIXED skips per-read
     *  pinning, answering zero or the kernel's negated @c errno, @c ENOMEM past the memlock cap. */
    long register_buffers(std::span<struct iovec const> regions) noexcept {
        assert(outstanding_ == 0 && "Registered buffers move only with no read in flight");
        u32_t const buffers = 0; // IORING_REGISTER_BUFFERS
        long const registered = register_(buffers, regions.data(), static_cast<unsigned>(regions.size()));
        if (registered < 0) return -static_cast<long>(errno);
        registered_regions_ = static_cast<u32_t>(regions.size());
        return 0;
    }

    /** Unpins the registered regions; a ring holding none asks the kernel nothing. */
    void unregister_buffers() noexcept {
        if (!registered_regions_) return;
        u32_t const unregister = 1; // IORING_UNREGISTER_BUFFERS
        [[maybe_unused]] long const unregistered = register_(unregister, nullptr, 0u);
        registered_regions_ = 0;
    }

    /** Registers @p descriptors as the ring's fixed-file table, addressed by
     *  @ref io_request_t::medium, removing per-read descriptor refcounting. */
    status_t register_files(std::span<int const> descriptors) noexcept {
        u32_t const files = 2; // IORING_REGISTER_FILES
        long const registered = register_(files, descriptors.data(), static_cast<unsigned>(descriptors.size()));
        return registered < 0 ? status_t::io_k : status_t::success_k;
    }

    void unregister_files() noexcept {
        u32_t const unregister = 3; // IORING_UNREGISTER_FILES
        [[maybe_unused]] long const unregistered = register_(unregister, nullptr, 0u);
    }

    /** Claims the next SQE slot, or returns null when the ring is full or holds its queue depth of
     *  reads. The caller fills the returned entry in place; @ref submit publishes it. The claim
     *  cursor is @c pending_tail_ rather than the kernel-visible tail, so a batch of claims arrives
     *  in consecutive slots before one publication. */
    linux_uring_sqe_t* next_submission() noexcept {
        if (outstanding_ == queue_depth_) return nullptr;
        u32_t const consumed = fu::standard_atomic_ref<u32_t>(*submissions_.head_word).load(std::memory_order_acquire);
        if (pending_tail_ - consumed == submissions_.mask + 1) return nullptr;
        u32_t const index = pending_tail_ & submissions_.mask;
        submissions_.order[index] = index;
        pending_tail_++;
        ++outstanding_;
        return reinterpret_cast<linux_uring_sqe_t*>(submissions_.entries + index * submissions_.stride);
    }

    /** The full submit-and-wait discipline shared by every ring engine: nonblocking submits &
     *  returns; blocking on an async-posting ring pays one capped @p wait_type_ monitor on the CQ
     *  tail before the blocking enter, so a completion arriving in that window costs zero syscalls,
     *  while deferred & polled rings fuse the pending batch and the wait into a single enter, and
     *  with nothing outstanding it returns at once. A refused enter withdraws what the kernel never
     *  consumed and waits out what it did, leaving nothing to poll. */
    template <typename wait_type_>
    status_t submit(io_wait_t wait) noexcept {
        status_t const submitted = enter_and_wait_<wait_type_>(wait);
        if (failed(submitted)) abandon_();
        return submitted;
    }

    /** Moves as many CQEs as @p completions holds, answering how many, each naming the buffer its
     *  @c user_data carried and its context from @p request_contexts; NVMe passthrough status
     *  decoding is the caller's job, since this layer only surfaces the primary result word. */
    std::size_t poll_completions(std::span<io_completion_t> completions,
                                 std::span<u64_t const> request_contexts) noexcept {
        u32_t reaped = *completions_.head_word;
        u32_t const tail = completion_tail_();
        std::size_t polled = 0;
        for (; reaped != tail && polled != completions.size(); ++reaped, ++polled) {
            linux_uring_cqe_t const* const entry = reinterpret_cast<linux_uring_cqe_t const*>(
                completions_.entries + (reaped & completions_.mask) * completions_.stride);
            io_buffer_id_t const buffer = static_cast<io_buffer_id_t>(entry->user_data);
            completions[polled] = io_completion_t {buffer, entry->res, request_contexts[buffer]};
        }
        fu::standard_atomic_ref<u32_t>(*completions_.head_word).store(reaped, std::memory_order_release);
        outstanding_ -= polled;
        return polled;
    }

  private:

    /** Publishes claimed SQEs & performs one @c io_uring_enter with the given completion demand;
     *  @p min_complete of zero submits without blocking. A registered ring, from Linux 5.18, is
     *  addressed by its index. */
    status_t enter_(u32_t min_complete) noexcept {
        if (pending_tail_ != *submissions_.tail_word)
            fu::standard_atomic_ref<u32_t>(*submissions_.tail_word).store(pending_tail_, std::memory_order_release);
        // Everything the kernel has not consumed, so a short or refused enter resubmits it next
        // time.
        u32_t const to_submit = pending_tail_ -
                                fu::standard_atomic_ref<u32_t>(*submissions_.head_word).load(std::memory_order_acquire);
        // Polled rings only harvest device completions inside the syscall, and deferred task-run
        // rings only post them there, so both must ask for events every time.
        bool const getevents = min_complete || posting_ == linux_uring_posting_t::deferred_k ||
                               reaping_ == linux_uring_reaping_t::polled_k;
        if (!to_submit && !getevents) return status_t::success_k;
        long const enter_syscall = 426 + linux_syscall_offset_k; // __NR_io_uring_enter

        u32_t const getevents_flag = 1u << 0;  // IORING_ENTER_GETEVENTS
        u32_t const registered_ring = 1u << 4; // IORING_ENTER_REGISTERED_RING
        u32_t enter_flags = getevents ? getevents_flag : 0u;
        int enter_descriptor = ring_descriptor_;
        if (registered_ring_index_ != unregistered_ring_k) {
            enter_flags |= registered_ring;
            enter_descriptor = static_cast<int>(registered_ring_index_);
        }
        long const entered = syscall(enter_syscall, enter_descriptor, to_submit, min_complete, enter_flags, nullptr,
                                     0ull);
        if (entered < 0 && errno != EINTR && errno != EBUSY && errno != EAGAIN) return status_t::io_k;
        return status_t::success_k;
    }

    /** One @c io_uring_register of @p opcode over @p count @p arguments. */
    long register_(u32_t opcode, void const* arguments, unsigned count) const noexcept {
        long const register_syscall = 427 + linux_syscall_offset_k; // __NR_io_uring_register
        return syscall(register_syscall, ring_descriptor_, opcode, arguments, count);
    }

    /** The body of @ref submit, before a refusal abandons the reads. */
    template <typename wait_type_>
    status_t enter_and_wait_(io_wait_t wait) noexcept {
        if (!outstanding_) return status_t::success_k;
        if (wait == io_wait_t::nonblocking_k || completion_tail_() != *completions_.head_word) return enter_(0);
        if (reaps_asynchronously_()) {
            if (status_t const submitted = enter_(0); failed(submitted)) return submitted;
            u32_t const observed = completion_tail_();
            wait_type_ {}(completions_.tail_word, observed, 0u, fu::wait_capped_t {});
            if (completion_tail_() != observed) return status_t::success_k;
        }
        return enter_(1);
    }

    /** Withdraws the SQEs the kernel never consumed, waits out the reads it did, and drops every
     *  CQE; the outstanding count keeps only the reads the kernel refused to wait for. */
    void abandon_() noexcept {
        u32_t const consumed = fu::standard_atomic_ref<u32_t>(*submissions_.head_word).load(std::memory_order_acquire);
        // Without SQPOLL the kernel reads the tail only inside `io_uring_enter`, so it may move
        // back.
        outstanding_ -= pending_tail_ - consumed;
        pending_tail_ = consumed;
        fu::standard_atomic_ref<u32_t>(*submissions_.tail_word).store(consumed, std::memory_order_release);
        for (;;) {
            u32_t const tail = completion_tail_();
            outstanding_ -= tail - *completions_.head_word;
            fu::standard_atomic_ref<u32_t>(*completions_.head_word).store(tail, std::memory_order_release);
            if (!outstanding_ || failed(enter_(static_cast<u32_t>(outstanding_)))) return;
        }
    }

    u32_t completion_tail_() const noexcept {
        return fu::standard_atomic_ref<u32_t>(*completions_.tail_word).load(std::memory_order_acquire);
    }

    /** True only when CQEs arrive in shared memory between syscalls: immediate posting under
     *  interrupt-driven reaping. Deferred task-run and IOPOLL rings post CQEs strictly inside
     *  @c io_uring_enter, so a user-space tail monitor can never observe them and the submit & wait
     *  pair fuses into one enter. */
    bool reaps_asynchronously_() const noexcept {
        return posting_ == linux_uring_posting_t::immediate_k && reaping_ == linux_uring_reaping_t::interrupted_k;
    }

    /** Registers the ring descriptor so every enter addresses it by index,
     *  skipping the per-syscall descriptor-table lookup. Silent no-op fallback on
     *  pre-5.18 kernels, where the registration opcode is simply rejected. */
    void register_ring_descriptor_() noexcept {
        linux_uring_rsrc_update_t ring_update {};
        ring_update.update_value = static_cast<u64_t>(ring_descriptor_);
        ring_update.offset = unregistered_ring_k; // The kernel assigns the index.
        u32_t const ring_fds = 20;                // IORING_REGISTER_RING_FDS
        long const registered = register_(ring_fds, &ring_update, 1u);
        if (registered >= 0) registered_ring_index_ = ring_update.offset;
    }

    int ring_descriptor_ = -1;
    linux_uring_reaping_t reaping_ = linux_uring_reaping_t::interrupted_k;
    linux_uring_posting_t posting_ = linux_uring_posting_t::immediate_k;
    linux_uring_mapping_t submission_mapping_;
    linux_uring_mapping_t completion_mapping_;
    linux_uring_mapping_t entries_mapping_;
    linux_uring_submissions_t submissions_;
    linux_uring_completions_t completions_;
    u32_t pending_tail_ = 0;
    u32_t registered_ring_index_ = unregistered_ring_k;

    /** Regions the registered-buffer table holds, zero while it holds none. */
    u32_t registered_regions_ = 0;
    std::size_t queue_depth_ = 0;
    std::size_t outstanding_ = 0;
};

/** The most bytes the kernel pins as one registered buffer. */
inline constexpr std::size_t linux_uring_registered_range_bytes_k = std::size_t {1} << 30;

/**
 *  @brief The staging region a ring engine reads into, registered as pieces of whole buffers under
 *      the kernel's 1 GiB cap per registered buffer, and what became of that registration.
 *
 *  Each ring engine owns one and keeps its region through it. The piece table grows only when the
 *  region regrows, at reserve time, so a warm read allocates nothing, and names its piece in the
 *  entry's @c buf_index.
 */
template <typename allocator_type_>
class linux_uring_registered_buffers {
  public:
    explicit linux_uring_registered_buffers(allocator_type_ const& allocator) noexcept
        : buffers_(allocator), regions_(allocator) {}

    /** Asks for registration as @p registration says, from the next region on. */
    void request(linux_uring_buffer_registration_t registration) noexcept { registration_ = registration; }

    /**
     *  @brief Cuts the region into @p count buffers of @p bytes each, and under @c fixed_k
     *      registers it with @p ring in pieces whenever it moves.
     *
     *  A pin past @c RLIMIT_MEMLOCK turns the registration to @c plain_k, visible through
     *  @ref registration, and every later read pins its buffer itself.
     *
     *  @par Statuses:
     *  - @c overflow_k The region's byte count overflows.
     *  - @c bad_alloc_k The piece table could not grow.
     *  - @c io_k The kernel refused the registration for another reason than its memlock cap.
     */
    status_t reserve(linux_uring_ring_t& ring, std::size_t count, std::size_t bytes) noexcept {
        if (buffers_.covers(count, bytes)) return buffers_.reserve(count, bytes);
        ring.unregister_buffers();
        if (status_t const grown = buffers_.reserve(count, bytes); failed(grown)) return grown;
        if (registration_ == linux_uring_buffer_registration_t::plain_k) return status_t::success_k;
        buffers_per_region_ = max_of<std::size_t>(linux_uring_registered_range_bytes_k / buffers_.stride(), 1);
        std::size_t const held = buffers_.capacity() / buffers_.stride();
        std::size_t const pieces = divide_round_up<std::size_t>(held, buffers_per_region_);
        if (status_t const sized = regions_.resize_uninitialized(pieces); failed(sized)) return sized;
        for (std::size_t piece = 0; piece != pieces; ++piece) {
            std::size_t const first = piece * buffers_per_region_;
            std::size_t const piece_buffers = min_of(buffers_per_region_, held - first);
            regions_[piece] = iovec {buffers_.buffer(static_cast<io_buffer_id_t>(first)),
                                     piece_buffers * buffers_.stride()};
        }
        long const registered = ring.register_buffers(to_span(regions_));
        if (registered == -ENOMEM) registration_ = linux_uring_buffer_registration_t::plain_k;
        else if (registered < 0) return status_t::io_k;
        return status_t::success_k;
    }

    /** Whether reads land in registered pieces, or at plain addresses the kernel pins per read. */
    linux_uring_buffer_registration_t registration() const noexcept { return registration_; }

    /** The registered piece @p buffer lies in, which a fixed read names as its @c buf_index. */
    u16_t piece_of(io_buffer_id_t buffer) const noexcept { return static_cast<u16_t>(buffer / buffers_per_region_); }

    byte_t* buffer(io_buffer_id_t buffer) const noexcept { return buffers_.buffer(buffer); }

  private:

    /** The staging buffers reads arrive in. */
    io_buffers<allocator_type_> buffers_;

    /** The pieces registered with the ring, whole buffers each, under the kernel's cap. */
    growing_buffer<struct iovec, allocator_type_> regions_;

    /** Buffers one registered piece holds, the last piece holding the rest. */
    std::size_t buffers_per_region_ = 1;

    /** Whether reads use the registered pieces, until the kernel refuses to pin them. */
    linux_uring_buffer_registration_t registration_ = linux_uring_buffer_registration_t::plain_k;
};

/**
 *  @brief io_uring reads against files and block devices: @c IORING_OP_READ_FIXED over buffers
 *      registered once, or @c IORING_OP_READ at plain addresses, with @c IORING_SETUP_IOPOLL when
 *      the config asks for polled queues.
 *
 *  Kernel floor 6.1 for the single-issuer deferred-task-run mode, degrading to a plain 5.x ring
 *  transparently. Pair with @c O_DIRECT media for the page-cache-free path the block-aligned reads
 *  are designed around. Registered buffers & files remove per-read page pinning & descriptor
 *  refcounting, roughly an 11% throughput gain at NVMe rates; polling harvests completions from the
 *  device's queues inside @c io_uring_enter instead of by interrupt, worth ~21% on top, and needs
 *  poll-capable queues: @c nvme.poll_queues=N for real drives, @c poll_queues in the null_blk
 *  recipe above for emulation; cross-check any setup with @c fio --ioengine=io_uring --hipri.
 *
 *  The submit discipline: submit everything claimed, then, on rings where completions post
 *  asynchronously, give the ForkUnion waiter one capped monitor on the CQ tail before paying the
 *  blocking syscall; a completion arriving within that ~microsecond window costs zero syscalls to
 *  poll, since the CQ is drained entirely in user space. Under @c DEFER_TASKRUN the spin is
 *  skipped, since completions only post inside @c io_uring_enter by design.
 */
template <typename wait_type_, typename allocator_type_ = forkunion_allocator_t>
class linux_uring_files_engine {
  public:
    using wait_t = wait_type_;
    using allocator_t = allocator_type_;
    using medium_t = file_medium_t;
    using config_t = io_engine_config_t;

    static constexpr io_transport_t transport() noexcept { return io_transport_t::linux_uring_k; }

    explicit linux_uring_files_engine(allocator_t const& allocator = {}) noexcept
        : buffers_(allocator), request_contexts_(allocator), descriptors_(allocator) {}
    linux_uring_files_engine(linux_uring_files_engine&&) noexcept = default;
    linux_uring_files_engine& operator=(linux_uring_files_engine&&) noexcept = default;

    /** Sets up the ring for @p config's depth, buffers and reaping; a refused @c io_uring_setup
     *  answers @c io_k. */
    status_t open(config_t const& config) noexcept {
        if (status_t const vetted = io_engine_vet_(config, transport()); failed(vetted)) return vetted;
        if (status_t const created = ring_.setup(config.queue_depth, linux_uring_form_t::compact_k, config.reaping);
            failed(created))
            return created;
        buffers_.request(config.buffer_registration);
        return status_t::success_k;
    }

    status_t reserve(std::size_t buffers, std::size_t buffer_bytes) noexcept {
        if (status_t const grown = buffers_.reserve(ring_, buffers, buffer_bytes); failed(grown)) return grown;
        return request_contexts_.size() >= buffers ? status_t::success_k : request_contexts_.resize(buffers);
    }

    /** Whether reads land in buffers registered once, or at addresses the kernel pins per read. */
    linux_uring_buffer_registration_t buffer_registration() const noexcept { return buffers_.registration(); }

    template <typename media_type_>
    status_t attach(std::span<media_type_ const> media) noexcept {
        detach();
        if (status_t const copied = copy_descriptors_(descriptors_, media); failed(copied)) return copied;
        if (descriptors_.empty()) return status_t::success_k;
        if (status_t const registered = ring_.register_files(descriptors_); failed(registered))
            return descriptors_.clear(), registered;
        return status_t::success_k;
    }

    void detach() noexcept {
        if (!descriptors_.empty()) ring_.unregister_files();
        descriptors_.clear();
    }

    status_t enqueue_read(io_request_t request) noexcept {
        u8_t const read = 22;      // IORING_OP_READ
        u8_t const read_fixed = 4; // IORING_OP_READ_FIXED
        if (status_t const vetted = io_request_vet_(request, descriptors_.size()); failed(vetted)) return vetted;
        linux_uring_sqe_t* const entry = ring_.next_submission();
        if (!entry) return status_t::overflow_k;
        *entry = linux_uring_sqe_t {};
        entry->opcode = read;
        entry->entry_flags = linux_uring_entry_fixed_file_k;
        // A fixed-file index, not a descriptor: the request's medium ID picks the file.
        entry->fd = static_cast<i32_t>(request.medium);
        entry->addr = reinterpret_cast<std::uintptr_t>(buffers_.buffer(request.buffer));
        entry->length = request.length;
        entry->offset = request.offset;
        entry->user_data = request.buffer;
        if (buffers_.registration() == linux_uring_buffer_registration_t::fixed_k)
            entry->opcode = read_fixed, entry->buf_index = buffers_.piece_of(request.buffer);
        request_contexts_[request.buffer] = request.context;
        return status_t::success_k;
    }

    status_t submit(io_wait_t wait) noexcept { return ring_.submit<wait_t>(wait); }

    std::size_t poll_completions(std::span<io_completion_t> completions) noexcept {
        return ring_.poll_completions(completions, to_span(request_contexts_));
    }

    std::byte* buffer(io_buffer_id_t buffer) noexcept { return buffers_.buffer(buffer); }

  private:

    /** The staging buffers reads arrive in, and their registration with the ring. */
    linux_uring_registered_buffers<allocator_t> buffers_;

    /** The context of the read each buffer serves, indexed by buffer. */
    growing_buffer<u64_t, allocator_t> request_contexts_;

    /** The attached media's descriptors, registered as the ring's fixed files. */
    growing_buffer<int, allocator_t> descriptors_;

    /** The ring, declared after the buffers so it dies first and releases its pins on them. */
    linux_uring_ring_t ring_;
};

/**
 *  @brief NVMe passthrough engine: raw Read commands to the namespace character device
 *      @c /dev/ngXnY through 128-byte SQEs, bypassing the filesystem and the block layer, worth
 *      ~20-30% IOPS per core over the block path.
 *
 *  Kernel floor 6.2 for unprivileged reads gated on the device node's file mode; a udev rule
 *  granting the service user read access to @c /dev/ng* suffices, no @c CAP_SYS_ADMIN. URING_CMD
 *  arrived in Linux 5.19 and its registered-buffer form, @c IORING_URING_CMD_FIXED, in 6.1.
 *
 *  The payload must live on the namespace directly: offsets translate to LBAs by a plain shift of
 *  each medium's @c block_bytes, so open the media with the namespace's LBA size in mind. Discover
 *  & reshape the geometry with @c nvme-cli:
 *
 *  @code{.sh}
 *  nvme id-ns /dev/nvme0n1           # `lbads:9` = 512e, `lbads:12` = 4Kn
 *  nvme format /dev/nvme0n1 --lbaf=0  # reformat, DESTROYS the namespace
 *  @endcode
 *
 *  Deployment notes: on trusted storage servers boot with @c iommu=pt so DMA skips per-read IOMMU
 *  mapping, avoiding a measured ~4x IOPS cliff on mapping-bound paths. Completion results surface
 *  NVMe status: zero is success, positives are NVMe status codes, negatives remain kernel errors.
 *  @c poll_completions folds both failure families into negative results and successes into the
 *  bytes read.
 */
template <typename wait_type_, typename allocator_type_ = forkunion_allocator_t>
class linux_uring_nvme_engine {
  public:
    using wait_t = wait_type_;
    using allocator_t = allocator_type_;
    using medium_t = file_medium_t;
    using config_t = io_engine_config_t;

    static constexpr io_transport_t transport() noexcept { return io_transport_t::linux_uring_nvme_k; }

    explicit linux_uring_nvme_engine(allocator_t const& allocator = {}) noexcept
        : buffers_(allocator), request_contexts_(allocator), lengths_(allocator), descriptors_(allocator),
          namespaces_(allocator) {}
    linux_uring_nvme_engine(linux_uring_nvme_engine&&) noexcept = default;
    linux_uring_nvme_engine& operator=(linux_uring_nvme_engine&&) noexcept = default;

    /** Sets up a 128-byte-entry ring for @p config's depth, buffers and reaping. */
    status_t open(config_t const& config) noexcept {
        if (status_t const vetted = io_engine_vet_(config, transport()); failed(vetted)) return vetted;
        if (status_t const created = ring_.setup(config.queue_depth, linux_uring_form_t::nvme_k, config.reaping);
            failed(created))
            return created;
        buffers_.request(config.buffer_registration);
        return status_t::success_k;
    }

    status_t reserve(std::size_t buffers, std::size_t buffer_bytes) noexcept {
        if (status_t const grown = buffers_.reserve(ring_, buffers, buffer_bytes); failed(grown)) return grown;
        // The lengths grow second, so a refusal between the two leaves them short and the next call
        // grows both.
        if (lengths_.size() >= buffers) return status_t::success_k;
        if (failed(request_contexts_.resize(buffers)) || failed(lengths_.resize(buffers))) return status_t::bad_alloc_k;
        return status_t::success_k;
    }

    /** Whether commands name buffers registered once, or addresses the kernel pins per read. */
    linux_uring_buffer_registration_t buffer_registration() const noexcept { return buffers_.registration(); }

    /**
     *  @brief Learns each medium's namespace ID and LBA shift, and registers every medium as a
     *      fixed file of the ring.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k A medium that is not a generic NVMe char node, like @c /dev/ng1n1.
     *  - @c io_k A char node that answers no namespace ID.
     */
    template <typename media_type_>
    status_t attach(std::span<media_type_ const> media) noexcept {
        detach();
        if (status_t const copied = copy_descriptors_(descriptors_, media); failed(copied)) return copied;
        if (status_t const reserved = namespaces_.reserve(media.size()); failed(reserved))
            return descriptors_.clear(), reserved;
        for (media_type_ const& medium : media) {
            file_medium_t const& file = *medium_as<file_medium_t>(medium);
            // Block nodes answer NVME_IOCTL_ID too, but only the generic char node takes URING_CMD.
            struct stat description;
            if (::fstat(file.descriptor, &description) != 0 || !S_ISCHR(description.st_mode))
                return descriptors_.clear(), namespaces_.clear(), status_t::config_mismatch_k;
            int const identifier = ::ioctl(file.descriptor, linux_nvme_ioctl_id_k);
            if (identifier < 0 || !file.block_bytes) return descriptors_.clear(), namespaces_.clear(), status_t::io_k;
            namespaces_.push_back_reserved(nvme_namespace_t {static_cast<u32_t>(identifier),
                                                             static_cast<u32_t>(std::countr_zero(file.block_bytes))});
        }
        if (descriptors_.empty()) return status_t::success_k;
        if (status_t const registered = ring_.register_files(descriptors_); failed(registered))
            return descriptors_.clear(), namespaces_.clear(), registered;
        return status_t::success_k;
    }

    void detach() noexcept {
        if (!descriptors_.empty()) ring_.unregister_files();
        descriptors_.clear();
        namespaces_.clear();
    }

    status_t enqueue_read(io_request_t request) noexcept {
        u8_t const uring_cmd = 46;                                           // IORING_OP_URING_CMD
        u32_t const uring_cmd_io = _IOWR('N', 0x80, linux_nvme_uring_cmd_t); // NVME_URING_CMD_IO
        u32_t const fixed = 1u << 0;                                         // IORING_URING_CMD_FIXED
        if (status_t const vetted = io_request_vet_(request, descriptors_.size()); failed(vetted)) return vetted;
        // A passthrough ring's slots are the 128-byte form.
        linux_uring_sqe128_t* const entry = reinterpret_cast<linux_uring_sqe128_t*>(ring_.next_submission());
        if (!entry) return status_t::overflow_k;
        *entry = linux_uring_sqe128_t {};
        entry->opcode = uring_cmd;
        entry->entry_flags = linux_uring_entry_fixed_file_k;
        entry->fd = static_cast<i32_t>(request.medium);
        entry->cmd_op = uring_cmd_io;
        entry->user_data = request.buffer;
        if (buffers_.registration() == linux_uring_buffer_registration_t::fixed_k)
            entry->uring_cmd_flags = fixed, entry->buf_index = buffers_.piece_of(request.buffer);
        nvme_namespace_t const& name_space = namespaces_[request.medium];
        linux_nvme_uring_cmd_t* const command = reinterpret_cast<linux_nvme_uring_cmd_t*>(entry->cmd);
        command->opcode = nvme_read_opcode_k;
        command->nsid = name_space.identifier;
        command->addr = reinterpret_cast<std::uintptr_t>(buffers_.buffer(request.buffer));
        command->data_length = request.length;
        u64_t const first_block = request.offset >> name_space.lba_shift;
        u32_t const block_count = request.length >> name_space.lba_shift;
        command->cdw10 = static_cast<u32_t>(first_block);
        command->cdw11 = static_cast<u32_t>(first_block >> 32);
        command->cdw12 = block_count - 1;
        request_contexts_[request.buffer] = request.context;
        lengths_[request.buffer] = request.length;
        return status_t::success_k;
    }

    status_t submit(io_wait_t wait) noexcept { return ring_.submit<wait_t>(wait); }

    /** Rewrites the NVMe convention, where zero means the full transfer, into byte counts. */
    std::size_t poll_completions(std::span<io_completion_t> completions) noexcept {
        std::size_t const polled = ring_.poll_completions(completions, to_span(request_contexts_));
        for (io_completion_t& completion : completions.first(polled))
            completion.result = completion.result == 0 ? static_cast<i32_t>(lengths_[completion.buffer])
                                                       : (completion.result > 0 ? -EIO : completion.result);
        return polled;
    }

    std::byte* buffer(io_buffer_id_t buffer) noexcept { return buffers_.buffer(buffer); }

  private:

    /** One attached namespace: its ID and the shift from bytes to its logical blocks. */
    struct nvme_namespace_t {

        /** The namespace ID @c NVME_IOCTL_ID reports. */
        u32_t identifier;

        /** log2 of the LBA size. */
        u32_t lba_shift;
    };

    /** The staging buffers commands land in, and their registration with the ring. */
    linux_uring_registered_buffers<allocator_t> buffers_;

    /** The context of the read each buffer serves, indexed by buffer. */
    growing_buffer<u64_t, allocator_t> request_contexts_;

    /** The bytes the command each buffer serves moves when it succeeds, since an NVMe completion
     *  reports a status rather than a byte count. */
    growing_buffer<u32_t, allocator_t> lengths_;

    /** The attached media's descriptors, registered as the ring's fixed files. */
    growing_buffer<int, allocator_t> descriptors_;

    /** The attached media's namespaces, indexed by medium ID. */
    growing_buffer<nvme_namespace_t, allocator_t> namespaces_;

    /** The ring, declared after the buffers so it dies first and releases its pins on them. */
    linux_uring_ring_t ring_;
};

#endif

#if USEARCH_WITH_WINDOWS_IORING

/**
 *  @brief Windows I/O Rings engine, the closest io_uring sibling: shared rings, user-mode
 *      completion polling, batched submission, and registered buffers & files mirroring the Linux
 *      fixed path.
 *
 *  Requires Windows 11 22H2 for @c IORING_VERSION_3; pair with @c FILE_FLAG_NO_BUFFERING handles,
 *  whose alignment contract matches this format: offsets, lengths & buffers aligned to the volume
 *  sector size, 512 or 4096, queried via @c FILE_STORAGE_INFO.
 *
 *  Registration ops are themselves asynchronous SQEs, so @c reserve and @c attach submit each and
 *  wait for its completion before any read can share the queue with it. Note WSL2 cannot substitute
 *  for this engine's testing: its storage arrives via a paravirtual SCSI path, so neither IoRing
 *  nor NVMe character devices exist there; only native Windows exercises this code.
 */
template <typename wait_type_, typename allocator_type_ = forkunion_allocator_t>
class windows_ioring_files_engine {
  public:
    using wait_t = wait_type_;
    using allocator_t = allocator_type_;
    using medium_t = file_medium_t;
    using config_t = io_engine_config_t;

    static constexpr io_transport_t transport() noexcept { return io_transport_t::windows_ioring_k; }

    explicit windows_ioring_files_engine(allocator_t const& allocator = {}) noexcept
        : buffers_(allocator), request_contexts_(allocator), descriptors_(allocator) {}
    windows_ioring_files_engine(windows_ioring_files_engine&& other) noexcept
        : buffers_(std::move(other.buffers_)), request_contexts_(std::move(other.request_contexts_)),
          descriptors_(std::move(other.descriptors_)), ring_(std::exchange(other.ring_, nullptr)),
          queue_depth_(other.queue_depth_), outstanding_(std::exchange(other.outstanding_, 0u)) {}
    windows_ioring_files_engine& operator=(windows_ioring_files_engine&& other) noexcept {
        buffers_ = std::move(other.buffers_);
        request_contexts_ = std::move(other.request_contexts_);
        descriptors_ = std::move(other.descriptors_);
        std::swap(ring_, other.ring_);
        std::swap(queue_depth_, other.queue_depth_);
        std::swap(outstanding_, other.outstanding_);
        return *this;
    }
    ~windows_ioring_files_engine() noexcept {
        if (ring_) ::CloseIoRing(ring_);
    }

    status_t open(config_t const& config) noexcept {
        if (status_t const vetted = io_engine_vet_(config, transport()); failed(vetted)) return vetted;
        IORING_CREATE_FLAGS const flags {IORING_CREATE_REQUIRED_FLAGS_NONE, IORING_CREATE_ADVISORY_FLAGS_NONE};
        // At most the queue depth of reads is outstanding, each leaving one completion.
        HRESULT const created = ::CreateIoRing(IORING_VERSION_3, flags, static_cast<UINT32>(config.queue_depth),
                                               static_cast<UINT32>(config.queue_depth), &ring_);
        if (!SUCCEEDED(created)) return status_t::io_k;
        queue_depth_ = config.queue_depth;
        return status_t::success_k;
    }

    /** Cuts the buffers from one region, registered as buffer zero whenever it moves, so reads
     *  address it by offset whatever stride a call picks. */
    status_t reserve(std::size_t buffers, std::size_t buffer_bytes) noexcept {
        if (request_contexts_.size() < buffers)
            if (status_t const grown = request_contexts_.resize(buffers); failed(grown)) return grown;
        if (buffers_.covers(buffers, buffer_bytes)) return buffers_.reserve(buffers, buffer_bytes);
        if (status_t const grown = buffers_.reserve(buffers, buffer_bytes); failed(grown)) return grown;
        if (buffers_.capacity() > ~UINT32 {0}) return buffers_.reset(), status_t::overflow_k;
        IORING_BUFFER_INFO const region {buffers_.base(), static_cast<UINT32>(buffers_.capacity())};
        HRESULT const built = ::BuildIoRingRegisterBuffers(ring_, 1, &region, 0);
        if (!SUCCEEDED(built) || failed(settle_())) return buffers_.reset(), status_t::io_k;
        return status_t::success_k;
    }

    template <typename media_type_>
    status_t attach(std::span<media_type_ const> media) noexcept {
        if (status_t const copied = copy_descriptors_(descriptors_, media); failed(copied)) return copied;
        if (descriptors_.empty()) return status_t::success_k;
        HRESULT const built = ::BuildIoRingRegisterFileHandles(ring_, static_cast<UINT32>(descriptors_.size()),
                                                               descriptors_.data(), 0);
        if (!SUCCEEDED(built) || failed(settle_())) return descriptors_.clear(), status_t::io_k;
        return status_t::success_k;
    }

    void detach() noexcept { descriptors_.clear(); }

    status_t enqueue_read(io_request_t request) noexcept {
        if (outstanding_ == queue_depth_) return status_t::overflow_k;
        if (status_t const vetted = io_request_vet_(request, descriptors_.size()); failed(vetted)) return vetted;
        IORING_HANDLE_REF const file = ::IoRingHandleRefFromIndex(request.medium);
        IORING_BUFFER_REF const destination = ::IoRingBufferRefFromIndexAndOffset(
            0, static_cast<UINT32>(static_cast<std::size_t>(request.buffer) * buffers_.stride()));
        HRESULT const built = ::BuildIoRingReadFile(ring_, file, destination, request.length, request.offset,
                                                    request.buffer, IOSQE_FLAGS_NONE);
        if (!SUCCEEDED(built)) return status_t::overflow_k;
        request_contexts_[request.buffer] = request.context;
        ++outstanding_;
        return status_t::success_k;
    }

    status_t submit(io_wait_t wait) noexcept {
        if (!outstanding_) return status_t::success_k;
        UINT32 const wanted = wait == io_wait_t::blocking_k ? 1u : 0u;
        UINT32 submitted = 0;
        if (SUCCEEDED(::SubmitIoRing(ring_, wanted, wanted ? INFINITE : 0u, &submitted))) return status_t::success_k;
        // Waits out every read the ring holds, then drops their completions.
        [[maybe_unused]] HRESULT const drained = ::SubmitIoRing(ring_, static_cast<UINT32>(outstanding_), INFINITE,
                                                                &submitted);
        IORING_CQE entry;
        while (::PopIoRingCompletion(ring_, &entry) == S_OK) {}
        outstanding_ = 0;
        return status_t::io_k;
    }

    std::size_t poll_completions(std::span<io_completion_t> completions) noexcept {
        std::size_t polled = 0;
        IORING_CQE entry;
        while (polled != completions.size() && ::PopIoRingCompletion(ring_, &entry) == S_OK) {
            io_buffer_id_t const buffer = static_cast<io_buffer_id_t>(entry.UserData);
            i32_t const result = SUCCEEDED(entry.ResultCode) ? static_cast<i32_t>(entry.Information) : -EIO;
            completions[polled++] = io_completion_t {buffer, result, request_contexts_[buffer]};
        }
        outstanding_ -= polled;
        return polled;
    }

    std::byte* buffer(io_buffer_id_t buffer) noexcept { return buffers_.buffer(buffer); }

  private:

    /** Submits a registration and waits for its completion, the only one in the queue. */
    status_t settle_() noexcept {
        UINT32 submitted = 0;
        IORING_CQE entry {};
        if (!SUCCEEDED(::SubmitIoRing(ring_, 1, INFINITE, &submitted)) || ::PopIoRingCompletion(ring_, &entry) != S_OK)
            return status_t::io_k;
        return SUCCEEDED(entry.ResultCode) ? status_t::success_k : status_t::io_k;
    }

    /** The staging buffers reads arrive in. */
    io_buffers<allocator_t> buffers_;

    /** The context of the read each buffer serves, indexed by buffer. */
    growing_buffer<u64_t, allocator_t> request_contexts_;

    /** The attached media's handles, registered with the ring. */
    growing_buffer<native_file_t, allocator_t> descriptors_;

    /** The ring, null until @c open. */
    HIORING ring_ = nullptr;

    /** Reads outstanding at most; zero until @c open. */
    std::size_t queue_depth_ = 0;

    /** Reads enqueued and not yet polled. */
    std::size_t outstanding_ = 0;
};

#endif

#pragma endregion Ring Engines

#pragma region SPDK Engine

#if USEARCH_WITH_LINUX_SPDK

/** Allocates DMA-able memory from the SPDK environment, which PCIe controllers write into
 *  directly, on one NUMA socket, unzeroed, since every read overwrites what it lands in. */
template <typename value_type_ = byte_t>
struct linux_spdk_dma_allocator {
    using value_type = value_type_;

    /** The NUMA socket the pages come from, @c SPDK_ENV_SOCKET_ID_ANY for any. */
    int socket = SPDK_ENV_SOCKET_ID_ANY;

    constexpr linux_spdk_dma_allocator() noexcept = default;

    /** Places on the socket of @p memory_domain, any socket for @c no_memory_domain_k. */
    explicit constexpr linux_spdk_dma_allocator(fu_memory_domain_id_t memory_domain) noexcept
        : socket(memory_domain == no_memory_domain_k ? SPDK_ENV_SOCKET_ID_ANY : static_cast<int>(memory_domain)) {}

    /** The rebinding conversion @c std::allocator_traits performs; the socket travels. */
    template <typename other_type_>
    constexpr linux_spdk_dma_allocator(linux_spdk_dma_allocator<other_type_> const& other) noexcept
        : socket(other.socket) {}

    value_type_* allocate(std::size_t count) noexcept {
        return static_cast<value_type_*>(
            spdk_dma_malloc_socket(count * sizeof(value_type_), io_alignment_k, nullptr, socket));
    }
    void deallocate(value_type_* pointer, std::size_t) noexcept { spdk_dma_free(pointer); }
    bool operator==(linux_spdk_dma_allocator const& other) const noexcept { return socket == other.socket; }
};

/**
 *  @brief The SPDK user-space driver: one queue pair per controller this thread reads, kept from
 *      first attach to @c release, DMA buffers from the environment, and polled completions.
 *
 *  Nothing here blocks, because the driver offers no interrupt at all:
 *  @c spdk_nvme_qpair_process_completions is the only way a command is ever seen to finish, so a
 *  blocking @c submit spins on it with the engine's waiter between polls. Queue pairs carry no
 *  lock, which is why each engine keeps its own and only its thread drives them. Commands are
 *  queued with delayed doorbells, so @c enqueue_read only encodes and the next poll rings once per
 *  batch.
 */
template <typename wait_type_, typename allocator_type_ = forkunion_allocator_t>
class linux_spdk_engine {
  public:
    using wait_t = wait_type_;
    using allocator_t = allocator_type_;
    using medium_t = spdk_medium_t;
    using config_t = io_engine_config_t;

    static constexpr io_transport_t transport() noexcept { return io_transport_t::linux_spdk_k; }

    explicit linux_spdk_engine(allocator_t const& allocator = {}) noexcept
        : buffers_(linux_spdk_dma_allocator<> {allocator.memory_domain}), queue_pairs_(allocator),
          namespaces_(allocator), commands_(allocator), request_contexts_(allocator), completions_(allocator) {}
    linux_spdk_engine(linux_spdk_engine&&) noexcept = default;
    linux_spdk_engine& operator=(linux_spdk_engine&&) noexcept = default;
    ~linux_spdk_engine() noexcept { assert(queue_pairs_.empty() && "An engine is released before it dies"); }

    /** Sizes the completion queue for @p config's depth, once the environment runs. */
    status_t open(config_t const& config) noexcept {
        if (status_t const vetted = io_engine_vet_(config, transport()); failed(vetted)) return vetted;
        if (!linux_spdk_process_t::running()) return status_t::uninitialized_k;
        if (failed(completions_.reserve_exactly(config.queue_depth))) return status_t::bad_alloc_k;
        queue_depth_ = config.queue_depth;
        return status_t::success_k;
    }

    status_t reserve(std::size_t buffers, std::size_t buffer_bytes) noexcept {
        if (status_t const grown = buffers_.reserve(buffers, buffer_bytes); failed(grown)) return grown;
        // The contexts grow second, so a refusal between the two leaves them short and the next
        // call grows both.
        if (request_contexts_.size() >= buffers) return status_t::success_k;
        if (failed(commands_.resize(buffers)) || failed(request_contexts_.resize(buffers)))
            return status_t::bad_alloc_k;
        return status_t::success_k;
    }

    /** Maps each medium to this thread's queue pair on its controller, allocating that queue pair
     *  once. */
    template <typename media_type_>
    status_t attach(std::span<media_type_ const> media) noexcept {
        namespaces_.clear();
        if (status_t const reserved = namespaces_.reserve(media.size()); failed(reserved)) return reserved;
        for (media_type_ const& medium : media) {
            spdk_medium_t const* const name_space = medium_as<spdk_medium_t>(medium);
            if (!name_space) return namespaces_.clear(), status_t::config_mismatch_k;
            auto const [queue_pair, status] = queue_pair_of_(*name_space->connection);
            if (failed(status)) return namespaces_.clear(), status;
            namespaces_.push_back_reserved(
                attached_namespace_t {name_space->name_space, queue_pair, name_space->block_bytes});
        }
        return status_t::success_k;
    }

    void detach() noexcept { namespaces_.clear(); }

    /** Frees every queue pair and its hold on the controller, on the engine's own thread. */
    void release() noexcept { release_queue_pairs_(); }

    status_t enqueue_read(io_request_t request) noexcept {
        if (outstanding_ == queue_depth_) return status_t::overflow_k;
        if (status_t const vetted = io_request_vet_(request, namespaces_.size()); failed(vetted)) return vetted;
        attached_namespace_t const& name_space = namespaces_[request.medium];
        request_contexts_[request.buffer] = request.context;
        // Commands address whole sectors, so a read off the grid fails alone, as `O_DIRECT` answers
        // it.
        if ((request.offset | request.length) % name_space.block_bytes) {
            complete_(request.buffer, -EINVAL);
            ++outstanding_;
            return status_t::success_k;
        }
        command_t& command = commands_[request.buffer];
        command = command_t {this, request.buffer, request.length};
        int const issued = spdk_nvme_ns_cmd_read(
            name_space.name_space, name_space.queue_pair, buffers_.buffer(request.buffer),
            request.offset / name_space.block_bytes, request.length / name_space.block_bytes,
            &linux_spdk_engine::completed_, &command, 0);
        if (issued == -ENOMEM) return status_t::overflow_k;
        // A command the driver refuses outright fails alone.
        if (issued != 0) complete_(request.buffer, issued);
        ++outstanding_;
        return status_t::success_k;
    }

    status_t submit(io_wait_t wait) noexcept {
        while (true) {
            for (queue_pair_t const& queue_pair : queue_pairs_)
                if (spdk_nvme_qpair_process_completions(queue_pair.queue_pair, 0) < 0) return abandon_();
            if (wait == io_wait_t::nonblocking_k || !completions_.empty() || !outstanding_) return status_t::success_k;
            wait_t {}(&outstanding_, outstanding_, 0u, fu::wait_capped_t {});
        }
    }

    std::size_t poll_completions(std::span<io_completion_t> completions) noexcept {
        std::size_t const polled = poll_queued_completions_(completions_, completions);
        outstanding_ -= polled;
        return polled;
    }

    std::byte* buffer(io_buffer_id_t buffer) noexcept { return buffers_.buffer(buffer); }

  private:

    /** This thread's queue pair on one controller. */
    struct queue_pair_t {

        /** The controller the queue pair belongs to and holds. */
        linux_spdk_process_t::connected_t* connection;

        /** The queue pair, freed on @c release. */
        spdk_nvme_qpair* queue_pair;
    };

    /** One attached medium, resolved to the queue pair that reads it. */
    struct attached_namespace_t {

        /** The namespace reads target. */
        spdk_nvme_ns* name_space;

        /** This thread's queue pair on the namespace's controller. */
        spdk_nvme_qpair* queue_pair;

        /** The namespace's sector size. */
        u32_t block_bytes;
    };

    /** What a command's callback receives: its engine, the buffer it arrives in, and its length. */
    struct command_t {

        /** The engine whose completion queue the callback fills. */
        linux_spdk_engine* engine;

        /** The buffer the command arrives in, which its completion names. */
        io_buffer_id_t buffer;

        /** The bytes a successful command moved. */
        u32_t length;
    };

    /** Queues the completion of the read @p buffer served, with @p result and its context. */
    void complete_(io_buffer_id_t buffer, i32_t result) noexcept {
        completions_.push_back_reserved(io_completion_t {buffer, result, request_contexts_[buffer]});
    }

    /** The driver's callback, run inside @c spdk_nvme_qpair_process_completions on this thread. */
    static void completed_(void* callback_argument, spdk_nvme_cpl const* completion) noexcept {
        command_t const& command = *static_cast<command_t const*>(callback_argument);
        i32_t const result = spdk_nvme_cpl_is_error(completion) ? -EIO : static_cast<i32_t>(command.length);
        command.engine->complete_(command.buffer, result);
    }

    /** This thread's queue pair on @p connected, allocated with delayed doorbells on first use. */
    expected<spdk_nvme_qpair*> queue_pair_of_(linux_spdk_process_t::connected_t& connected) noexcept {
        for (queue_pair_t const& queue_pair : queue_pairs_)
            if (queue_pair.connection == &connected) return {queue_pair.queue_pair, status_t::success_k};
        spdk_nvme_ctrlr* const controller = connected.controller;
        spdk_nvme_io_qpair_opts options;
        spdk_nvme_ctrlr_get_default_io_qpair_opts(controller, &options, sizeof(options));
        options.io_queue_size = max_of<u32_t>(options.io_queue_size, static_cast<u32_t>(queue_depth_ + 1));
        // SPDK's floor; a split read past the pool waits as an overflow for earlier reads.
        options.io_queue_requests = max_of<u32_t>(options.io_queue_requests, options.io_queue_size);
        options.delay_cmd_submit = true;
        spdk_nvme_qpair* const allocated = spdk_nvme_ctrlr_alloc_io_qpair(controller, &options, sizeof(options));
        if (!allocated) return {nullptr, status_t::io_k};
        if (status_t const kept = queue_pairs_.push_back(queue_pair_t {&connected, allocated}); failed(kept))
            return spdk_nvme_ctrlr_free_io_qpair(allocated), expected<spdk_nvme_qpair*> {nullptr, kept};
        linux_spdk_process_t::hold<wait_t>(connected);
        return {allocated, status_t::success_k};
    }

    /** Frees every queue pair; the driver completes what they held through @c completed_. */
    void release_queue_pairs_() noexcept {
        for (queue_pair_t const& queue_pair : queue_pairs_) {
            spdk_nvme_ctrlr_free_io_qpair(queue_pair.queue_pair);
            linux_spdk_process_t::release<wait_t>(*queue_pair.connection);
        }
        queue_pairs_.clear();
        namespaces_.clear();
    }

    /** Frees the failed queue pairs, which aborts what they held, and empties the engine. */
    status_t abandon_() noexcept {
        release_queue_pairs_();
        completions_.clear();
        outstanding_ = 0;
        return status_t::io_k;
    }

    /** The DMA staging buffers reads arrive in. */
    io_buffers<linux_spdk_dma_allocator<>> buffers_;

    /** One queue pair per controller this thread has read. */
    growing_buffer<queue_pair_t, allocator_t> queue_pairs_;

    /** The attached media, indexed by medium ID. */
    growing_buffer<attached_namespace_t, allocator_t> namespaces_;

    /** The command each buffer serves, which its callback receives, indexed by buffer. */
    growing_buffer<command_t, allocator_t> commands_;

    /** The context of the read each buffer serves, indexed by buffer. */
    growing_buffer<u64_t, allocator_t> request_contexts_;

    /** Reads finished and not yet polled. */
    growing_buffer<io_completion_t, allocator_t> completions_;

    /** Reads outstanding at most; zero until @c open. */
    std::size_t queue_depth_ = 0;

    /** Reads enqueued and not yet polled. */
    std::size_t outstanding_ = 0;
};

#endif

#pragma endregion SPDK Engine

#pragma region Engine Sets

/** Folds @p types_ into a @c std::variant of the distinct ones in order, or the lone type when they
 *  agree. */
template <typename kept_type_, typename... types_>
struct distinct_media_ {
    using type = kept_type_;
};

template <typename only_type_>
struct distinct_media_<std::variant<only_type_>> {
    using type = only_type_;
};

template <typename... kept_types_, typename next_type_, typename... types_>
struct distinct_media_<std::variant<kept_types_...>, next_type_, types_...> {
    using type = typename distinct_media_<
        std::conditional_t<(std::is_same_v<kept_types_, next_type_> || ...), std::variant<kept_types_...>,
                           std::variant<kept_types_..., next_type_>>,
        types_...>::type;
};

/**
 *  @brief Several engines behind one type: @c open picks one at runtime, and every concept member
 *      forwards to it through one @c std::visit, a predictable switch per call.
 *
 *  An @c automatic_k configuration tries the kernel files engines in order, so io_uring refused by
 *  a seccomp profile falls to AIO, and a refused AIO context to blocking reads. Any other transport
 *  opens exactly its own engine or answers why it could not. The media are a @c std::variant of the
 *  engines' distinct media types, or the one type they share.
 */
template <typename... engine_types_>
class io_engine_variant {
  public:
    using engines_t = std::variant<engine_types_...>;
    using allocator_t = typename std::variant_alternative_t<0, engines_t>::allocator_t;
    using medium_t = typename distinct_media_<std::variant<>, typename engine_types_::medium_t...>::type;
    using config_t = io_engine_config_t;

    static_assert((std::is_same_v<typename engine_types_::allocator_t, allocator_t> && ...),
                  "Every engine of a set places its buffers through one allocator");

    explicit io_engine_variant(allocator_t const& allocator = {}) noexcept
        : engines_(std::in_place_index<0>, allocator), allocator_(allocator) {}

    /**
     *  @brief Opens the engine @p config names, or under @c automatic_k the first kernel files
     *      engine whose setup the host admits, a refused setup passing to the next.
     *
     *  @par Statuses:
     *  - @c unsupported_k The set has no engine for the named transport, or none set up.
     *  - @c io_k The named engine's setup was refused.
     *  - @c invalid_argument_k A zero queue depth.
     *  - @c uninitialized_k SPDK named before its environment started.
     */
    status_t open(config_t const& config) noexcept { return open_<0>(config); }

    /** The transport of the engine @c open picked. */
    io_transport_t transport() const noexcept {
        return std::visit([](auto const& engine) noexcept { return engine.transport(); }, engines_);
    }

    /**
     *  Cuts the picked engine's staging region into @p buffers buffers of @p buffer_bytes each.
     *
     *  @par Statuses:
     *  - @c overflow_k The region's byte count overflows, or passes what Windows registers.
     */
    status_t reserve(std::size_t buffers, std::size_t buffer_bytes) noexcept {
        return std::visit([&](auto& engine) noexcept { return engine.reserve(buffers, buffer_bytes); }, engines_);
    }

    /**
     *  Attaches @p media to the picked engine, which request medium IDs index until @c detach.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k A medium the picked engine cannot read.
     */
    status_t attach(std::span<medium_t const> media) noexcept {
        return std::visit([&](auto& engine) noexcept { return engine.attach(media); }, engines_);
    }

    void detach() noexcept {
        std::visit([](auto& engine) noexcept { engine.detach(); }, engines_);
    }

    /** Gives back what the engine holds on devices other engines share, as SPDK queue pairs do. */
    void release() noexcept {
        std::visit(
            [](auto& engine) noexcept {
                if constexpr (requires { engine.release(); }) engine.release();
            },
            engines_);
    }

    /**
     *  Queues @p request on the picked engine for the next @c submit.
     *
     *  @par Statuses:
     *  - @c overflow_k The queue already holds its depth.
     *  - @c invalid_argument_k A medium never attached, or a length past what a completion counts.
     */
    status_t enqueue_read(io_request_t request) noexcept {
        return std::visit([&](auto& engine) noexcept { return engine.enqueue_read(request); }, engines_);
    }

    /** Submits every queued request on the picked engine, waiting as @p wait says. */
    status_t submit(io_wait_t wait) noexcept {
        return std::visit([&](auto& engine) noexcept { return engine.submit(wait); }, engines_);
    }

    std::size_t poll_completions(std::span<io_completion_t> completions) noexcept {
        return std::visit([&](auto& engine) noexcept { return engine.poll_completions(completions); }, engines_);
    }

    std::byte* buffer(io_buffer_id_t buffer) noexcept {
        return std::visit([&](auto& engine) noexcept { return engine.buffer(buffer); }, engines_);
    }

    /** Whether the picked engine reads into buffers registered once; @c plain_k for one that
     *  registers none. */
    linux_uring_buffer_registration_t buffer_registration() const noexcept {
        return std::visit(
            [](auto const& engine) noexcept {
                if constexpr (requires { engine.buffer_registration(); }) return engine.buffer_registration();
                else return linux_uring_buffer_registration_t::plain_k;
            },
            engines_);
    }

  private:
    template <std::size_t index_>
        requires(index_ == sizeof...(engine_types_))
    status_t open_(config_t const&) noexcept {
        return status_t::unsupported_k;
    }

    template <std::size_t index_>
        requires(index_ < sizeof...(engine_types_))
    status_t open_(config_t const& config) noexcept {
        using engine_t = std::variant_alternative_t<index_, engines_t>;
        if (!opens_(engine_t::transport(), config.transport)) return open_<index_ + 1>(config);
        status_t const opened = engines_.template emplace<index_>(allocator_).open(config);
        // A named transport opens exactly or answers why; only the automatic ladder degrades.
        if (opened != status_t::io_k || config.transport != io_transport_t::automatic_k) return opened;
        return open_<index_ + 1>(config);
    }

    /** Whether an engine driving @p engine answers a request for @p wanted: the named one alone,
     *  or under @c automatic_k every kernel files engine. */
    static constexpr bool opens_(io_transport_t engine, io_transport_t wanted) noexcept {
        if (wanted == io_transport_t::automatic_k)
            return engine != io_transport_t::linux_spdk_k && engine != io_transport_t::linux_uring_nvme_k;
        return engine == wanted;
    }

    engines_t engines_;
    [[no_unique_address]] allocator_t allocator_;
};

/** The engine a set materializes: the one engine itself, so a single-engine build dispatches
 *  nothing, or an @c io_engine_variant over several. */
template <typename... engine_types_>
using any_io_engine =
    std::conditional_t<sizeof...(engine_types_) == 1, std::variant_alternative_t<0, std::variant<engine_types_...>>,
                       io_engine_variant<engine_types_...>>;

/** Every engine this build admits, each waiting through @p wait_type_, in the order a kernel path
 *  tries them; blocking reads close every list, and passthrough opens only when named. */
template <typename wait_type_>
using default_io_engine = any_io_engine<
#if USEARCH_WITH_LINUX_URING
    linux_uring_nvme_engine<wait_type_>, linux_uring_files_engine<wait_type_>,
#endif
#if USEARCH_WITH_LINUX_AIO
    linux_aio_files_engine<wait_type_>,
#endif
#if USEARCH_WITH_LINUX_SPDK
    linux_spdk_engine<wait_type_>,
#endif
#if USEARCH_WITH_WINDOWS_IORING
    windows_ioring_files_engine<wait_type_>,
#endif
    blocking_files_engine<wait_type_>>;

#pragma endregion Engine Sets

#pragma region Executor

/**
 *  @brief The threads a storage walk advances on, each with its scratch arena and its own engine.
 *
 *  Composes @c forkunion_executor_t for the threads, their arenas and the one-caller hold, and
 *  builds each engine on its own thread, so a ring's single issuer is the thread that submits to
 *  it. A call attaches a view's media to every engine for its own duration and detaches on return,
 *  so no state outlives a call and any executor serves any view.
 */
template <typename engine_type_>
class nvme_executor {
  public:
    using engine_t = engine_type_;
    using medium_t = typename engine_t::medium_t;
    using config_t = typename engine_t::config_t;
    using allocator_t = typename engine_t::allocator_t;
    using arena_t = forkunion_executor_t::arena_t;
    using claim_t = forkunion_executor_t::claim_t;

    static_assert(io_engine<engine_t>, "An executor drives engines through the I/O engine concept");

    /** One call's media, attached on every thread until it dies. */
    class attachment_t {
      public:
        attachment_t() noexcept = default;
        attachment_t(attachment_t&& other) noexcept : executor_(std::exchange(other.executor_, nullptr)) {}
        attachment_t& operator=(attachment_t&& other) noexcept {
            std::swap(executor_, other.executor_);
            return *this;
        }
        attachment_t(attachment_t const&) = delete;
        attachment_t& operator=(attachment_t const&) = delete;
        ~attachment_t() noexcept {
            if (executor_) executor_->detach_();
        }

      private:
        friend class nvme_executor;
        explicit attachment_t(nvme_executor* executor) noexcept : executor_(executor) {}

        /** The executor whose engines hold the media, null once moved out. */
        nvme_executor* executor_ = nullptr;
    };

    nvme_executor() noexcept = default;
    nvme_executor(nvme_executor&&) noexcept = default;
    nvme_executor& operator=(nvme_executor&&) noexcept = default;

    ~nvme_executor() noexcept { close_engines_(); }

    /** Releases every engine on its own thread, which must precede the executor's death once an
     *  engine holds devices other engines share. */
    void release() noexcept {
        if constexpr (requires(engine_t& engine) { engine.release(); }) {
            [[maybe_unused]] status_t const released = threads_.for_threads(
                [&](std::size_t thread, std::size_t) noexcept {
                    if (thread < engines_.size()) engines_[thread].release();
                });
        }
    }

    /**
     *  @brief Spawns @p threads over @p topology under the ForkUnion facilities @p allowed names,
     *      zero threads meaning @c forkunion_executor_t::default_threads, on the compute domains
     *      local to @p near_domain, or across the machine without one.
     *
     *  Each engine opens with @p config on its own thread, every one on the transport thread zero's
     *  engine picked. A nonzero @p config queue depth pins every call's depth, while zero leaves
     *  each call to derive its own from the media it reads, the engines one read deep until then.
     *
     *  @par Statuses:
     *  - @c uninitialized_k SPDK is named without a driver.
     *  - @c unsupported_k The build has no engine for @p config's transport, or none set up.
     *  - @c io_k The named engine's setup was refused.
     */
    static expected<nvme_executor> spawn(fu_topology_t topology, std::size_t threads, config_t const& config,
                                         fu_capabilities_t allowed = fu_capabilities_all_k,
                                         fu_memory_domain_id_t near_domain = no_memory_domain_k) noexcept {
        nvme_executor executor;
        // The caller never runs a slice, so each thread index stays one OS thread, the single
        // issuer a ring wants.
        bool const os_threads = fu_comptime_capabilities() & fu_capability_os_threads_k;
        auto [threads_executor, spawn_status] =
            os_threads && near_domain != no_memory_domain_k
                ? forkunion_executor_t::spawn_near(topology, near_domain, threads, allowed)
                : forkunion_executor_t::spawn(topology, threads,
                                              os_threads ? fu_caller_exclusive_k : fu_caller_inclusive_k, allowed);
        executor.threads_ = std::move(threads_executor);
        if (failed(spawn_status)) return {std::move(executor), spawn_status};
        if (status_t const sized = executor.engines_.resize(executor.threads_.size()); failed(sized))
            return {std::move(executor), sized};
        executor.pinned_depth_ = config.queue_depth;
        // An unpinned depth waits for the media; one read deep picks the transport.
        executor.opened_ = config, executor.opened_.queue_depth = max_of<std::size_t>(config.queue_depth, 1);
        // A fallback decided per thread could mix transports, so the others follow thread zero's
        // choice or fail.
        status_t opened = executor.on_every_thread_([&](std::size_t thread) noexcept {
            return thread ? status_t::success_k : executor.open_engine_(thread, executor.opened_);
        });
        if (failed(opened)) return {std::move(executor), opened};
        executor.opened_.transport = executor.engines_[0].transport();
        opened = executor.on_every_thread_([&](std::size_t thread) noexcept {
            return thread ? executor.open_engine_(thread, executor.opened_) : status_t::success_k;
        });
        return {std::move(executor), opened};
    }

    /**
     *  Claims the executor for one call.
     *
     *  @par Statuses:
     *  - @c contended_k Another call holds the executor.
     */
    expected<claim_t> claim() noexcept { return threads_.claim(); }

    /**
     *  @brief Reopens every engine @p queue_depth reads deep under @p reaping, on the transport
     *      they hold, when they were opened to another shape.
     *
     *  Runs with nothing in flight, and the buffers are reserved afresh after it.
     *
     *  @par Statuses:
     *  - @c contended_k The pool is running another dispatch.
     *  - @c io_k A thread's setup was refused at that shape.
     */
    status_t fit(std::size_t queue_depth, linux_uring_reaping_t reaping) noexcept {
        if (queue_depth == opened_.queue_depth && reaping == opened_.reaping) return status_t::success_k;
        config_t fitted = opened_;
        fitted.queue_depth = queue_depth, fitted.reaping = reaping;
        // Rings share a memlock limit with pinned buffers, so all engines close before any reopens.
        close_engines_();
        status_t const opened = on_every_thread_(
            [&](std::size_t thread) noexcept { return open_engine_(thread, fitted); });
        // A refusal may leave some engines reopened, so no shape is held and the next call reopens.
        opened_.queue_depth = succeeded(opened) ? queue_depth : 0;
        opened_.reaping = reaping;
        return opened;
    }

    /**
     *  Rewinds and seeds every thread's arena at @p scratch_bytes_per_thread, and cuts every
     *  engine's staging region into @p buffers buffers of @p buffer_bytes each, regrowing and
     *  re-registering it only when they do not fit.
     *
     *  @par Statuses:
     *  - @c uninitialized_k The executor holds no threads.
     *  - @c contended_k The pool is running another dispatch.
     *  - @c overflow_k A staging region's byte count overflows, or passes what Windows registers.
     */
    status_t reserve(std::size_t scratch_bytes_per_thread, std::size_t buffers, std::size_t buffer_bytes) noexcept {
        if (status_t const reserved = threads_.reserve(scratch_bytes_per_thread); failed(reserved)) return reserved;
        return reserve_buffers(buffers, buffer_bytes);
    }

    /**
     *  @brief Cuts every engine's staging region into @p buffers buffers of @p buffer_bytes each,
     *      as @ref reserve does, with nothing in flight, while the arenas keep what a call carved.
     *
     *  @par Statuses:
     *  - @c contended_k The pool is running another dispatch.
     *  - @c overflow_k A staging region's byte count overflows, or passes what Windows registers.
     */
    status_t reserve_buffers(std::size_t buffers, std::size_t buffer_bytes) noexcept {
        status_t const cut = on_every_thread_(
            [&](std::size_t thread) noexcept { return engines_[thread].reserve(buffers, buffer_bytes); });
        cut_ = succeeded(cut) ? io_buffers_cut_t {buffers, buffer_bytes} : io_buffers_cut_t {};
        return cut;
    }

    /** The buffers every engine's staging region was last cut into, and their bytes; zeros after a
     *  refused cut. */
    io_buffers_cut_t buffers_cut() const noexcept { return cut_; }

    /**
     *  Attaches @p media to every engine, so request medium IDs index them until the attachment
     *  dies; a refusal on any thread detaches them all.
     *
     *  @par Statuses:
     *  - @c uninitialized_k The executor holds no pool, or the pool has no workers.
     *  - @c contended_k The pool is running another dispatch.
     *  - @c config_mismatch_k A medium the engines cannot read.
     */
    expected<attachment_t> attach(std::span<medium_t const> media) noexcept {
        status_t const attached = on_every_thread_(
            [&](std::size_t thread) noexcept { return engines_[thread].attach(media); });
        if (failed(attached)) return detach_(), expected<attachment_t> {attachment_t {}, attached};
        return {attachment_t {this}, status_t::success_k};
    }

    /** Thread @p thread 's engine, driven only from that thread. */
    engine_t& engine(std::size_t thread) noexcept { return engines_[thread]; }

    /** Thread @p thread 's arena; no two threads share one. */
    arena_t& arena(std::size_t thread) noexcept { return threads_.arena(thread); }

    /**
     *  Runs @p callback on every thread with the thread index and its compute domain; must be
     *  @c noexcept.
     *
     *  @par Statuses:
     *  - @c uninitialized_k The executor holds no pool, or the pool has no workers.
     *  - @c contended_k The pool is running another dispatch.
     */
    template <typename callback_type_>
    status_t for_threads(callback_type_&& callback) noexcept {
        return threads_.for_threads(std::forward<callback_type_>(callback));
    }

    /**
     *  Runs @p callback over @p tasks items, stolen dynamically, with the task index, the thread
     *  index and its compute domain; must be @c noexcept.
     *
     *  @par Statuses:
     *  - @c uninitialized_k The executor holds no pool, or the pool has no workers.
     *  - @c contended_k The pool is running another dispatch.
     */
    template <typename callback_type_>
    status_t for_tasks(std::size_t tasks, callback_type_&& callback) noexcept {
        return threads_.for_tasks(tasks, std::forward<callback_type_>(callback));
    }

    /** Nothing is ever pending: every fan-out blocks inside, so a drain is a no-op. */
    status_t wait() noexcept { return threads_.wait(); }

    /** The pool, its per-domain arenas and its one-caller hold. */
    forkunion_executor_t& threads() noexcept { return threads_; }

    /** Threads this executor drives. */
    std::size_t size() const noexcept { return threads_.size(); }

    /** Reads each engine keeps outstanding at most: the pinned depth, or the last call's. */
    std::size_t queue_depth() const noexcept { return opened_.queue_depth; }

    /** The depth the spawn's config pinned for every call, zero when each call derives its own. */
    std::size_t pinned_depth() const noexcept { return pinned_depth_; }

    /** Whether every engine reads into buffers registered once: @c plain_k once any kernel refused
     *  to pin its thread's region, or where the engines register none. */
    linux_uring_buffer_registration_t buffer_registration() const noexcept {
        if constexpr (requires(engine_t const& engine) { engine.buffer_registration(); }) {
            for (engine_t const& engine : engines_)
                if (engine.buffer_registration() == linux_uring_buffer_registration_t::plain_k)
                    return linux_uring_buffer_registration_t::plain_k;
            return engines_.empty() ? linux_uring_buffer_registration_t::plain_k
                                    : linux_uring_buffer_registration_t::fixed_k;
        }
        return linux_uring_buffer_registration_t::plain_k;
    }

  private:

    /** Builds thread @p thread 's engine on its domain's allocator and opens it with @p config;
     *  runs on that thread. */
    status_t open_engine_(std::size_t thread, config_t const& config) noexcept {
        engine_t& engine = engines_[thread];
        engine = engine_t {allocator_t {threads_.memory_domain(thread)}};
        return engine.open(config);
    }

    /** Runs @p callback on every thread, answering the first failure any of them reported. */
    template <typename callback_type_>
    status_t on_every_thread_(callback_type_&& callback) noexcept {
        status_t failure = status_t::success_k;
        status_t const dispatched = threads_.for_threads([&](std::size_t thread, std::size_t) noexcept {
            keep_first_failure<fu::standard_atomic_ref>(failure, callback(thread));
        });
        return failed(dispatched) ? dispatched : failure;
    }

    /** Replaces every engine with an unopened one on its own thread, the only one a single-issuer
     *  ring takes an unregistration from. */
    void close_engines_() noexcept {
        [[maybe_unused]] status_t const closed = threads_.for_threads([&](std::size_t thread, std::size_t) noexcept {
            if (thread < engines_.size()) engines_[thread] = engine_t {allocator_t {threads_.memory_domain(thread)}};
        });
    }

    /** Detaches every engine, each on its own thread. */
    void detach_() noexcept {
        [[maybe_unused]] status_t const detached = threads_.for_threads(
            [&](std::size_t thread, std::size_t) noexcept { engines_[thread].detach(); });
    }

    /** The pool, the per-domain arenas and the one-caller hold. */
    forkunion_executor_t threads_;

    /** The shape every engine was opened to: its transport, depth and reaping. */
    config_t opened_;

    /** The depth the spawn pinned, zero when calls derive theirs. */
    std::size_t pinned_depth_ = 0;

    /** What @ref reserve_buffers last cut every engine's staging region into. */
    io_buffers_cut_t cut_ {};

    /** One engine per thread, built and driven only on its thread; declared last, so engines die
     *  first. */
    growing_buffer<engine_t> engines_ {global_allocator};
};

#pragma endregion Executor

} // namespace unum::usearch::utilities
