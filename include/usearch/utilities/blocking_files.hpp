/**
 *  @file include/usearch/utilities/blocking_files.hpp
 *  @author Ash Vardanian
 *  @date September 4, 2026
 *  @brief Files as bytes, two ways: @c seekable_file_t addressed by offset with sliceable proxies
 *      over its contents, and @c mapped_file_t addressed as memory - so a reader and a writer
 *      both speak @c subspan and @c copy_bytes wherever the bytes sit.
 *
 *  Every verb here blocks the calling thread, which is the whole difference from the NVMe
 *  engines: those queue their reads and return, where these park until the kernel answers.
 *  Bulk work - a whole checkpoint, a whole dataset - is what this is for, and it needs no
 *  @c io_uring, no I/O Rings and no @c O_DIRECT, so it builds and runs everywhere.
 *
 *  The format this backs is memory-mappable by design, so reading is a mapping and every reader
 *  works on @c bytes_view_t. Writing has two shapes for one reason: a mapping dirties the page
 *  cache, which a terabyte checkpoint should not, so @ref unum::usearch::utilities::file_span_t
 *  addresses a descriptor positionally and @ref unum::usearch::utilities::copy_bytes puts the
 *  difference in an overload rather than at a call site. Slices carry their own absolute offset,
 *  which is what an aligned direct write will want.
 *
 *  Both stand on one set of native file verbs, @ref unum::usearch::utilities::open_native_file and
 *  its read, write, flush and close, which the NVMe media open through too, so each platform's
 *  difference is spelled once. A small file that must never tear, like a manifest, is replaced
 *  whole through @ref unum::usearch::utilities::replace_file_durably.
 */
#pragma once

#include <cerrno>  // `errno`, `EINTR`, `ENOENT`, `ENOSPC`
#include <cstddef> // `std::byte`
#include <cstdio>  // `std::rename`, `std::remove` - a durable replacement's move and its cleanup
#include <cstring> // `std::memcpy`, `std::memset`

#include <charconv>     // `std::from_chars`, for the keys of a file address
#include <memory>       // `std::allocator`, `std::allocator_traits`
#include <span>         // `std::span`
#include <string_view>  // `std::string_view`
#include <system_error> // `std::errc`
#include <utility>      // `std::exchange`

#include "usearch/utilities/types.hpp"

/*  `<winsock2.h>` goes first to keep Winsock 1.1 out, unless a user's earlier `<windows.h>` already
 *  brought it in. */
#if USEARCH_OS_WINDOWS_ && !defined(_WINSOCKAPI_)
#include <winsock2.h>
#endif
#if USEARCH_OS_WINDOWS_
#include <windows.h> // `CreateFileA`, `CreateFileMappingA`, `MapViewOfFile`, `MoveFileExA`, `GetStdHandle`
#else
#include <fcntl.h>    // `open`, `fcntl`, `O_DIRECT`, `F_NOCACHE`, `F_FULLFSYNC`
#include <sys/stat.h> // `fstat`, `stat`, `S_ISBLK`, `S_ISREG`
#include <unistd.h>   // `pread`, `pwrite`, `read`, `write`, `ftruncate`, `fsync`, `fdatasync`, `close`
#endif
#if USEARCH_OS_LINUX_
#include <sys/ioctl.h> // `ioctl`
#include <sys/mount.h> // `BLKGETSIZE64`
#endif
#if !USEARCH_OS_WINDOWS_ && !USEARCH_OS_WASI_
#include <dlfcn.h>    // `dlopen`, `dlsym` - GPU runtimes opened at first use
#include <sys/mman.h> // `mmap`, `munmap`, `msync`
#endif

namespace unum::usearch::utilities {

#pragma region Native Files

/** How @ref open_native_file opens a file. */
enum class file_access_t {

    /** Reads an existing file, sharing it with other readers. */
    read_only_k,

    /** Reads and writes an existing file, keeping its bytes. */
    read_write_k,

    /** Creates the file, or empties the one there, for reading and writing. */
    truncate_read_write_k,

    /** Reads and writes an existing disk no mounted filesystem or other writer holds. */
    exclusive_read_write_k,
};

/** Whether @ref open_native_file reads and writes through the page cache or past it. */
enum class file_buffering_t {

    /** Through the page cache, any offset and length. */
    buffered_k,

    /** Past it where the platform can: @c O_DIRECT, @c FILE_FLAG_NO_BUFFERING, or Darwin's
     *  @c F_NOCACHE hint, which unaligned transfers still serve from the cache. */
    unbuffered_k,
};

/** The most bytes one native transfer moves: a GiB, a multiple of every block an unbuffered file
 *  aligns to, and under Darwin's @c INT_MAX and Windows' @c DWORD caps. */
inline constexpr std::size_t native_transfer_bytes_k = std::size_t {1} << 30;

#if USEARCH_OS_WINDOWS_

/** What the platform calls an open file: a handle, overlapped, so every thread moves its own
 *  offset. */
using native_file_t = HANDLE;

/** The handle no open file carries. */
inline native_file_t const invalid_native_file_k = INVALID_HANDLE_VALUE;

/** The event one thread waits its positional transfers out on, so threads sharing a file never
 *  wake on each other's transfers; empty when Windows refused to create it. */
class native_file_event_t {
  public:
    native_file_event_t() noexcept : event_(::CreateEventA(nullptr, TRUE, FALSE, nullptr)) {}
    native_file_event_t(native_file_event_t&& other) noexcept : event_(std::exchange(other.event_, nullptr)) {}
    native_file_event_t& operator=(native_file_event_t&& other) noexcept {
        std::swap(event_, other.event_);
        return *this;
    }
    native_file_event_t(native_file_event_t const&) = delete;
    native_file_event_t& operator=(native_file_event_t const&) = delete;
    ~native_file_event_t() noexcept {
        if (event_) ::CloseHandle(event_);
    }

    /** The event's handle, null when Windows refused to create it. */
    HANDLE handle() const noexcept { return event_; }

  private:
    HANDLE event_ = nullptr;
};

#else

/** What the platform calls an open file: a descriptor. */
using native_file_t = int;

/** The descriptor no open file carries. */
inline constexpr native_file_t invalid_native_file_k = -1;

/** Nothing: a positional transfer blocks in its own syscall, so no thread waits on another's. */
struct native_file_event_t {};

/** The @c open flags of @p access, before buffering. */
constexpr int open_flags_of_(file_access_t access) noexcept {
    return access == file_access_t::read_only_k              ? O_RDONLY
           : access == file_access_t::truncate_read_write_k  ? O_RDWR | O_CREAT | O_TRUNC
           : access == file_access_t::exclusive_read_write_k ? O_RDWR | O_EXCL
                                                             : O_RDWR;
}

#endif // USEARCH_OS_WINDOWS_

#if USEARCH_OS_WINDOWS_

/** Opens @p path under @p access and @p buffering, or answers @c invalid_native_file_k. */
inline native_file_t open_native_file(char const* path, file_access_t access, file_buffering_t buffering) noexcept {
    DWORD const wanted = access == file_access_t::read_only_k ? GENERIC_READ : GENERIC_READ | GENERIC_WRITE;
    // As on POSIX, others may truncate, rename or delete an open file; only a claimed disk refuses.
    DWORD const shared = access == file_access_t::exclusive_read_write_k
                             ? 0
                             : FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE;
    DWORD const disposition = access == file_access_t::truncate_read_write_k ? CREATE_ALWAYS : OPEN_EXISTING;
    DWORD const unbuffered = buffering == file_buffering_t::unbuffered_k ? FILE_FLAG_NO_BUFFERING : 0;
    // Overlapped handles let every thread read one file at once, and I/O Rings ask for them.
    return ::CreateFileA(path, wanted, shared, nullptr, disposition, FILE_FLAG_OVERLAPPED | unbuffered, nullptr);
}

/** Makes every write so far durable, as far as the volume reports it so. */
inline status_t flush_native_file(native_file_t file) noexcept {
    return ::FlushFileBuffers(file) ? status_t::success_k : status_t::io_k;
}

#elif USEARCH_OS_APPLE_

/** Opens @p path under @p access and @p buffering, or answers @c invalid_native_file_k. */
inline native_file_t open_native_file(char const* path, file_access_t access, file_buffering_t buffering) noexcept {
    native_file_t const file = ::open(path, open_flags_of_(access) | O_CLOEXEC, 0644);
    // A hint rather than `O_DIRECT`: unaligned transfers fall back to the cache, and pages already
    // resident keep serving.
    if (file != invalid_native_file_k && buffering == file_buffering_t::unbuffered_k) ::fcntl(file, F_NOCACHE, 1);
    return file;
}

/** Makes every write so far durable past the drive's cache, or as far as @c fsync reaches where
 *  the filesystem has no such barrier. */
inline status_t flush_native_file(native_file_t file) noexcept {
    if (::fcntl(file, F_FULLFSYNC) == 0) return status_t::success_k;
    return errno == ENOTSUP && ::fsync(file) == 0 ? status_t::success_k : status_t::io_k;
}

#elif USEARCH_OS_LINUX_ || USEARCH_OS_FREEBSD_

/** Opens @p path under @p access and @p buffering, or answers @c invalid_native_file_k. */
inline native_file_t open_native_file(char const* path, file_access_t access, file_buffering_t buffering) noexcept {
    int const unbuffered = buffering == file_buffering_t::unbuffered_k ? O_DIRECT : 0;
    return ::open(path, open_flags_of_(access) | unbuffered | O_CLOEXEC, 0644);
}

/** Makes every write so far durable, without waiting on metadata a read does not need. */
inline status_t flush_native_file(native_file_t file) noexcept {
    return ::fdatasync(file) == 0 ? status_t::success_k : status_t::io_k;
}

#else

/** Opens @p path under @p access, or answers @c invalid_native_file_k; no flag here skips the page
 *  cache, so every file is buffered. */
inline native_file_t open_native_file(char const* path, file_access_t access, file_buffering_t) noexcept {
    return ::open(path, open_flags_of_(access) | O_CLOEXEC, 0644);
}

/** Makes every write so far durable, without waiting on metadata a read does not need. */
inline status_t flush_native_file(native_file_t file) noexcept {
    return ::fdatasync(file) == 0 ? status_t::success_k : status_t::io_k;
}

#endif // USEARCH_OS_WINDOWS_

#if USEARCH_OS_LINUX_

/** Bytes the block device @p descriptor spans, zero when the kernel does not say. */
inline u64_t block_device_bytes_(int descriptor) noexcept {
    u64_t bytes = 0;
    return ::ioctl(descriptor, BLKGETSIZE64, &bytes) == 0 ? bytes : 0;
}

/** The logical block of the block device @p descriptor, once its writes go out @c O_DIRECT; zero
 *  for anything else, which keeps writing through the page cache. */
inline std::size_t direct_write_block_bytes_(int descriptor) noexcept {
    struct stat description;
    int sector_bytes = 0;
    if (::fstat(descriptor, &description) != 0 || !S_ISBLK(description.st_mode) ||
        ::ioctl(descriptor, BLKSSZGET, &sector_bytes) != 0 || sector_bytes <= 0)
        return 0;
    int const flags = ::fcntl(descriptor, F_GETFL);
    if (flags < 0 || ::fcntl(descriptor, F_SETFL, flags | O_DIRECT) != 0) return 0;
    return static_cast<std::size_t>(sector_bytes);
}

#elif !USEARCH_OS_WINDOWS_

inline u64_t block_device_bytes_(int) noexcept { return 0; }

inline std::size_t direct_write_block_bytes_(int) noexcept { return 0; }

#endif

#if USEARCH_OS_WINDOWS_

/** Closes @p file, surfacing a failure the kernel reports. */
inline status_t close_native_file(native_file_t file) noexcept {
    return ::CloseHandle(file) ? status_t::success_k : status_t::io_k;
}

/** Reads @p destination at @p offset of @p file, waiting on @p event, answering the bytes moved,
 *  fewer only where the file ends or a pipe holds no more yet, or a negated @c errno. */
inline i64_t read_native_file(native_file_t file, native_file_event_t const& event, bytes_span_t destination,
                              u64_t offset) noexcept {
    if (!event.handle()) return -EIO;
    std::size_t moved = 0;
    while (moved != destination.size()) {
        OVERLAPPED overlapped {};
        overlapped.Offset = static_cast<DWORD>(offset + moved);
        overlapped.OffsetHigh = static_cast<DWORD>((offset + moved) >> 32);
        overlapped.hEvent = event.handle();
        DWORD const asked = static_cast<DWORD>(min_of(destination.size() - moved, native_transfer_bytes_k));
        DWORD transferred = 0;
        if ((!::ReadFile(file, destination.data() + moved, asked, nullptr, &overlapped) &&
             ::GetLastError() != ERROR_IO_PENDING) ||
            !::GetOverlappedResult(file, &overlapped, &transferred, TRUE)) {
            DWORD const error = ::GetLastError();
            // A pipe whose writer closed ends like a file.
            if (error == ERROR_HANDLE_EOF || error == ERROR_BROKEN_PIPE) break;
            return -EIO;
        }
        moved += transferred;
        // A file reads short only at its end, where an unbuffered one refuses the unaligned retry.
        if (transferred != asked) break;
    }
    return static_cast<i64_t>(moved);
}

/** Writes all of @p source at @p offset of @p file, waiting on @p event, answering the bytes
 *  moved or a negated @c errno, @c ENOSPC once the volume is full. */
inline i64_t write_native_file(native_file_t file, native_file_event_t const& event, bytes_view_t source,
                               u64_t offset) noexcept {
    if (!event.handle()) return -EIO;
    std::size_t moved = 0;
    while (moved != source.size()) {
        OVERLAPPED overlapped {};
        overlapped.Offset = static_cast<DWORD>(offset + moved);
        overlapped.OffsetHigh = static_cast<DWORD>((offset + moved) >> 32);
        overlapped.hEvent = event.handle();
        DWORD const asked = static_cast<DWORD>(min_of(source.size() - moved, native_transfer_bytes_k));
        DWORD transferred = 0;
        if ((!::WriteFile(file, source.data() + moved, asked, nullptr, &overlapped) &&
             ::GetLastError() != ERROR_IO_PENDING) ||
            !::GetOverlappedResult(file, &overlapped, &transferred, TRUE))
            return ::GetLastError() == ERROR_DISK_FULL ? -ENOSPC : -EIO;
        if (!transferred) return -EIO;
        moved += transferred;
    }
    return static_cast<i64_t>(moved);
}

/** Bytes @p file spans. */
inline expected<u64_t> native_file_capacity(native_file_t file) noexcept {
    LARGE_INTEGER size {};
    if (!::GetFileSizeEx(file, &size)) return {0, status_t::io_k};
    return {static_cast<u64_t>(size.QuadPart), status_t::success_k};
}

/** The process's standard input, which a pipe or a console reads in order, ignoring offsets. */
inline native_file_t standard_input_native_file() noexcept { return ::GetStdHandle(STD_INPUT_HANDLE); }

/** The process's standard output, which a pipe or a console writes in order, ignoring offsets. */
inline native_file_t standard_output_native_file() noexcept { return ::GetStdHandle(STD_OUTPUT_HANDLE); }

/** Whether @p path names a regular file or nothing at all, rather than a directory or a device. */
inline bool names_regular_file_or_nothing_(char const* path) noexcept {
    DWORD const attributes = ::GetFileAttributesA(path);
    return attributes == INVALID_FILE_ATTRIBUTES || !(attributes & (FILE_ATTRIBUTE_DIRECTORY | FILE_ATTRIBUTE_DEVICE));
}

#else

/** Closes @p file, surfacing a failure the kernel reports. */
inline status_t close_native_file(native_file_t file) noexcept {
    return ::close(file) == 0 ? status_t::success_k : status_t::io_k;
}

/** Reads @p destination at @p offset of @p file, answering the bytes moved, fewer only where the
 *  file ends, or a negated @c errno. */
inline i64_t read_native_file(native_file_t file, native_file_event_t const&, bytes_span_t destination,
                              u64_t offset) noexcept {
    std::size_t moved = 0;
    while (moved != destination.size()) {
        std::size_t const asked = min_of(destination.size() - moved, native_transfer_bytes_k);
        ssize_t transferred = ::pread(file, destination.data() + moved, asked, static_cast<off_t>(offset + moved));
        // A pipe or a terminal has no offsets, and reads in order.
        if (transferred < 0 && errno == ESPIPE) transferred = ::read(file, destination.data() + moved, asked);
        if (transferred < 0 && errno == EINTR) continue;
        if (transferred < 0) return -static_cast<i64_t>(errno);
        if (transferred == 0) break;
        moved += static_cast<std::size_t>(transferred);
    }
    return static_cast<i64_t>(moved);
}

/** Writes all of @p source at @p offset of @p file, answering the bytes moved or a negated
 *  @c errno, @c ENOSPC once the volume is full. */
inline i64_t write_native_file(native_file_t file, native_file_event_t const&, bytes_view_t source,
                               u64_t offset) noexcept {
    std::size_t moved = 0;
    while (moved != source.size()) {
        std::size_t const asked = min_of(source.size() - moved, native_transfer_bytes_k);
        ssize_t transferred = ::pwrite(file, source.data() + moved, asked, static_cast<off_t>(offset + moved));
        // A pipe or a terminal has no offsets, and writes in order.
        if (transferred < 0 && errno == ESPIPE) transferred = ::write(file, source.data() + moved, asked);
        if (transferred < 0 && errno == EINTR) continue;
        if (transferred < 0) return -static_cast<i64_t>(errno);
        if (transferred == 0) return -EIO;
        moved += static_cast<std::size_t>(transferred);
    }
    return static_cast<i64_t>(moved);
}

/** Bytes @p file spans: a regular file's length, or a block device's where the kernel says. */
inline expected<u64_t> native_file_capacity(native_file_t file) noexcept {
    struct stat description;
    if (::fstat(file, &description) != 0) return {0, status_t::io_k};
    if (S_ISBLK(description.st_mode)) return {block_device_bytes_(file), status_t::success_k};
    return {static_cast<u64_t>(description.st_size), status_t::success_k};
}

/** The process's standard input, which a pipe or a terminal reads in order, ignoring offsets. */
inline native_file_t standard_input_native_file() noexcept { return STDIN_FILENO; }

/** The process's standard output, which a pipe or a terminal writes in order, ignoring offsets. */
inline native_file_t standard_output_native_file() noexcept { return STDOUT_FILENO; }

/** Whether @p path names a regular file or nothing at all, rather than a directory or a device. */
inline bool names_regular_file_or_nothing_(char const* path) noexcept {
    struct stat description;
    return ::stat(path, &description) == 0 ? S_ISREG(description.st_mode) : errno == ENOENT;
}

#endif // USEARCH_OS_WINDOWS_

#pragma endregion Native Files

#pragma region Descriptors

/**
 *  @brief Where a part lives: the file, device node or SPDK namespace that holds it, and where its
 *      byte zero sits there. @ref parse_file_address reads it from any of these forms:
 *
 *  @verbatim
 *  /data/sift.safegraphs                              a file at byte zero
 *  /dev/nvme1n1@64GiB                                 a device node from byte 64 GiB
 *  /dev/nvme1n1@64GiB+16GiB                           the 16 GiB of it from byte 64 GiB on
 *  spdk://0000:5e:00.0@64GiB                          namespace 1 of a local PCIe controller
 *  spdk://0000:5e:00.0/2@64GiB                        namespace 2 of it
 *  trtype:PCIe traddr:0000:5e:00.0 ns:2 base:4096     an SPDK transport ID with its keys
 *  @endverbatim
 *
 *  A base and a length are whole numbers of B, KiB, MiB, GiB or TiB, all binary, bytes when bare;
 *  a file whose name ends in such a suffix takes an explicit @c @0 after it. A length bounds what a
 *  writer may size the part to.
 */
struct file_address_t {

    /** A file, a device node, the PCIe address of an @c spdk:// controller, or the tokens of an
     *  SPDK transport ID. */
    std::string_view location;

    /** The namespace an SPDK transport ID names, zero for its first active one. */
    u32_t namespace_id = 0;

    /** Where the part's byte zero sits on the medium, a multiple of its block. */
    u64_t byte_base = 0;

    /** Bytes past the base a writer may size the part to, zero meaning to the medium's end. */
    u64_t byte_limit = 0;
};

/** Reads @p token as @p key and a whole decimal number into @p value, left as it was otherwise. */
template <typename value_type_>
bool parse_address_key_(std::string_view token, std::string_view key, value_type_& value) noexcept {
    value_type_ parsed = 0;
    if (!token.starts_with(key)) return false;
    std::from_chars_result const result = std::from_chars(token.data() + key.size(), token.data() + token.size(),
                                                          parsed);
    if (result.ec != std::errc {} || result.ptr != token.data() + token.size()) return false;
    return value = parsed, true;
}

/** Reads @p text, a whole decimal number and an optional binary unit, into @p bytes, left as it
 *  was when the unit is unknown or the count overflows. */
inline bool parse_byte_count_(std::string_view text, u64_t& bytes) noexcept {
    u64_t count = 0;
    std::from_chars_result const result = std::from_chars(text.data(), text.data() + text.size(), count);
    if (result.ec != std::errc {}) return false;
    std::string_view const suffix = text.substr(static_cast<std::size_t>(result.ptr - text.data()));
    std::string_view const unit = suffix.empty() ? "B" : suffix;
    constexpr std::string_view units[] = {"B", "KiB", "MiB", "GiB", "TiB"};
    unsigned shift = 0;
    for (std::string_view const known : units) {
        if (unit == known) {
            if (count > ~u64_t {0} >> shift) return false;
            return bytes = count << shift, true;
        }
        shift += 10;
    }
    return false;
}

/**
 *  @brief Splits @p path into the parts of a @ref unum::usearch::utilities::file_address_t, in any
 *      form it lists.
 *
 *  @par Statuses:
 *  - @c invalid_argument_k The base or the length is in an unknown unit or overflows.
 *  - @c invalid_argument_k The namespace after an @c spdk:// controller is unreadable.
 *  - @c invalid_argument_k The path names no location.
 */
inline expected<file_address_t> parse_file_address(std::string_view path) noexcept {
    file_address_t parsed;
    std::string_view location = path.substr(0, path.find_last_not_of(" \t") + 1);
    for (std::size_t space = location.find_last_of(" \t"); space != std::string_view::npos;
         space = location.find_last_of(" \t")) {
        std::string_view const token = location.substr(space + 1);
        if (!parse_address_key_(token, "ns:", parsed.namespace_id) &&
            !parse_address_key_(token, "base:", parsed.byte_base))
            break;
        location = location.substr(0, location.find_last_not_of(" \t", space) + 1);
    }
    std::size_t const at = location.rfind('@');
    if (at != std::string_view::npos && at + 1 < location.size() && location[at + 1] >= '0' &&
        location[at + 1] <= '9') {
        std::string_view const extent = location.substr(at + 1);
        std::size_t const plus = extent.find('+');
        if ((plus != std::string_view::npos && !parse_byte_count_(extent.substr(plus + 1), parsed.byte_limit)) ||
            !parse_byte_count_(extent.substr(0, plus), parsed.byte_base))
            return {parsed, status_t::invalid_argument_k};
        location = location.substr(0, at);
    }
    std::string_view const scheme = "spdk://";
    if (location.starts_with(scheme)) {
        location.remove_prefix(scheme.size());
        std::size_t const slash = location.find('/');
        parsed.namespace_id = 1;
        if (slash != std::string_view::npos) {
            if (!parse_address_key_(location.substr(slash + 1), "", parsed.namespace_id))
                return {parsed, status_t::invalid_argument_k};
            location = location.substr(0, slash);
        }
    }
    parsed.location = location;
    return {parsed, location.empty() ? status_t::invalid_argument_k : status_t::success_k};
}

/** An owned file descriptor with positional verbs only - no cursor, so any thread reads or
 *  writes any offset; every verb answers @c status_t. */
struct seekable_file_t {
    seekable_file_t() = default;
    seekable_file_t(seekable_file_t&& other) noexcept
        : descriptor_(std::exchange(other.descriptor_, invalid_native_file_k)),
          byte_base_(std::exchange(other.byte_base_, 0u)), byte_limit_(std::exchange(other.byte_limit_, 0u)) {}
    seekable_file_t& operator=(seekable_file_t&& other) noexcept {
        std::swap(descriptor_, other.descriptor_);
        std::swap(byte_base_, other.byte_base_);
        std::swap(byte_limit_, other.byte_limit_);
        return *this;
    }
    seekable_file_t(seekable_file_t const&) = delete;
    seekable_file_t& operator=(seekable_file_t const&) = delete;
    ~seekable_file_t() noexcept { [[maybe_unused]] status_t const closed = close(); }

    /** Truthy exactly while a file is open. */
    explicit operator bool() const noexcept { return descriptor_ != invalid_native_file_k; }

    /**
     *  @brief Opens @p path, a file or a device in any form @ref file_address_t lists, every
     *      offset counting from its base.
     *
     *  The NUL terminator the OS demands is one transient allocation through @p allocator. A write
     *  past byte zero opens an existing medium, keeping the bytes before it, and a write to a disk
     *  opens it exclusively, so a mounted filesystem or another writer refuses it.
     *
     *  @par Statuses:
     *  - @c invalid_argument_k The path does not parse.
     */
    template <typename allocator_type_ = std::allocator<byte_t>>
    static expected<seekable_file_t> open(std::string_view path, file_access_t access,
                                          allocator_type_ allocator = {}) noexcept {
        auto const [parsed, parse_status] = parse_file_address(path);
        if (failed(parse_status)) return {seekable_file_t {}, parse_status};
        using chars_allocator_t = typename std::allocator_traits<allocator_type_>::template rebind_alloc<char>;
        chars_allocator_t chars {allocator};
        std::size_t const length = parsed.location.size();
        char* terminated = chars.allocate(length + 1);
        if (!terminated) return {seekable_file_t {}, status_t::bad_alloc_k};
        std::memcpy(terminated, parsed.location.data(), length);
        terminated[length] = '\0';
        seekable_file_t file;
        file.descriptor_ = open_(terminated, access, parsed.byte_base);
        file.byte_base_ = parsed.byte_base;
        file.byte_limit_ = parsed.byte_limit;
        chars.deallocate(terminated, length + 1);
        status_t const opened = file ? status_t::success_k : status_t::io_k;
        return {std::move(file), opened};
    }

    /**
     *  @brief Sets the file's length to @p bytes past its base, so positional writes land in a
     *      pre-sized file; a device keeps its size and answers @c overflow_k when they do not fit.
     *
     *  @par Statuses:
     *  - @c overflow_k A device ends before its base plus @p bytes.
     *  - @c overflow_k @p bytes passes the length the address bounded the part to.
     */
    status_t truncate(std::size_t bytes) noexcept;

    /** Writes every byte of @p source at @p offset, looping over short writes. */
    status_t write_at(std::size_t offset, bytes_view_t source) noexcept;

    /**
     *  Reads exactly @p destination.size() bytes at @p offset; a short read answers @c io_k.
     *
     *  @par Statuses:
     *  - @c io_k The read reaches the end of the file first.
     */
    status_t read_at(std::size_t offset, bytes_span_t destination) const noexcept;

    /** Byte length of the file or the device past its base. */
    expected<std::size_t> length() const noexcept;

    /** Makes every write so far durable, so a later write can be ordered after them. */
    status_t flush() noexcept { return flush_native_file(descriptor_); }

    /** Closes, surfacing the failure the destructor would swallow. */
    status_t close() noexcept {
        if (descriptor_ == invalid_native_file_k) return status_t::success_k;
        return close_native_file(std::exchange(descriptor_, invalid_native_file_k));
    }

    /** Hands the descriptor to a caller that closes it from now on - what a mapping takes. */
    native_file_t release() noexcept { return std::exchange(descriptor_, invalid_native_file_k); }

    /** Where offset zero sits in the file, which a mapping starts at. */
    u64_t byte_base() const noexcept { return byte_base_; }

    /** The descriptor, still owned here, for a driver that registers the file. */
    native_file_t descriptor() const noexcept { return descriptor_; }

  private:
    friend class mapped_file_t;

    /** Opens @p path under @p access for a part starting @p byte_base bytes in, answering
     *  @c invalid_native_file_k on failure. */
    static native_file_t open_(char const* path, file_access_t access, u64_t byte_base) noexcept {
        file_access_t opened = access;
        if (access != file_access_t::read_only_k && !names_regular_file_or_nothing_(path))
            opened = file_access_t::exclusive_read_write_k;
        else if (access != file_access_t::read_only_k && byte_base) opened = file_access_t::read_write_k;
        return open_native_file(path, opened, file_buffering_t::buffered_k);
    }

    /** The wrapped descriptor; @c invalid_native_file_k marks the empty state a move-out leaves
     *  behind. */
    native_file_t descriptor_ = invalid_native_file_k;

    /** Where offset zero sits in the file. */
    u64_t byte_base_ = 0;

    /** Bytes past the base @ref truncate may size the file to, zero meaning to the medium's end. */
    u64_t byte_limit_ = 0;
};

inline status_t seekable_file_t::write_at(std::size_t offset, bytes_view_t source) noexcept {
    i64_t const moved = write_native_file(descriptor_, native_file_event_t {}, source, byte_base_ + offset);
    return moved == static_cast<i64_t>(source.size()) ? status_t::success_k : status_t::io_k;
}

inline status_t seekable_file_t::read_at(std::size_t offset, bytes_span_t destination) const noexcept {
    i64_t const moved = read_native_file(descriptor_, native_file_event_t {}, destination, byte_base_ + offset);
    return moved == static_cast<i64_t>(destination.size()) ? status_t::success_k : status_t::io_k;
}

inline expected<std::size_t> seekable_file_t::length() const noexcept {
    auto const [bytes, capacity_status] = native_file_capacity(descriptor_);
    if (failed(capacity_status)) return {0, capacity_status};
    return {static_cast<std::size_t>(bytes - min_of(bytes, byte_base_)), status_t::success_k};
}

#if USEARCH_OS_WINDOWS_

inline status_t seekable_file_t::truncate(std::size_t bytes) noexcept {
    if (byte_limit_ && bytes > byte_limit_) return status_t::overflow_k;
    LARGE_INTEGER distance;
    distance.QuadPart = static_cast<LONGLONG>(byte_base_ + bytes);
    if (!::SetFilePointerEx(descriptor_, distance, nullptr, FILE_BEGIN)) return status_t::io_k;
    return ::SetEndOfFile(descriptor_) ? status_t::success_k : status_t::io_k;
}

#else

inline status_t seekable_file_t::truncate(std::size_t bytes) noexcept {
    if (byte_limit_ && bytes > byte_limit_) return status_t::overflow_k;
    struct stat description;
    if (::fstat(descriptor_, &description) != 0) return status_t::io_k;
    if (S_ISBLK(description.st_mode))
        return byte_base_ + bytes <= block_device_bytes_(descriptor_) ? status_t::success_k : status_t::overflow_k;
    return ::ftruncate(descriptor_, static_cast<off_t>(byte_base_ + bytes)) == 0 ? status_t::success_k : status_t::io_k;
}

#endif // USEARCH_OS_WINDOWS_

/** A shared library opened by path and kept open for the process's life, as the entry points
 *  resolved from it are never unloaded; empty where the system has no such library or, as on
 *  WASI, no loader. */
struct shared_library_t {
    void* handle = nullptr;

    /** Opens the library at @p path, or answers the empty one. */
    static shared_library_t open([[maybe_unused]] char const* path) noexcept {
#if USEARCH_OS_WINDOWS_
        return {reinterpret_cast<void*>(LoadLibraryA(path))};
#elif USEARCH_OS_WASI_
        return {};
#else
        return {dlopen(path, RTLD_NOW | RTLD_LOCAL)};
#endif
    }

    /** Points @p entry at the symbol exported as @p name, answering whether there is one. */
    template <typename function_type_>
    bool resolve([[maybe_unused]] char const* name, function_type_& entry) const noexcept {
        if (!handle) return false;
#if USEARCH_OS_WINDOWS_
        entry = reinterpret_cast<function_type_>(GetProcAddress(static_cast<HMODULE>(handle), name));
#elif !USEARCH_OS_WASI_
        entry = reinterpret_cast<function_type_>(dlsym(handle, name));
#endif
        return entry != nullptr;
    }

    explicit operator bool() const noexcept { return handle != nullptr; }
};

#pragma endregion Descriptors

#pragma region Slices and Mappings

/** A read-only run of a file's bytes: the descriptor and an absolute byte range, sliced
 *  like a span and read through @ref unum::usearch::utilities::copy_bytes. */
struct file_view_t {

    /** The file the range lives in; null for an empty slice. */
    seekable_file_t const* file = nullptr;

    /** First byte of the range within the file. */
    std::size_t begin = 0;

    /** One past the last byte. */
    std::size_t end = 0;

    std::size_t size() const noexcept { return end - begin; }
    bool empty() const noexcept { return end == begin; }

    /** The @p length bytes at @p offset within this range; empty when they do not fit. */
    file_view_t subspan(std::size_t offset, std::size_t length) const noexcept {
        if (offset > size() || length > size() - offset) return {};
        return {file, begin + offset, begin + offset + length};
    }
};

/** The writable twin of @ref file_view_t, written through
 *  @ref unum::usearch::utilities::copy_bytes. */
struct file_span_t {

    /** The file the range lives in; null for an empty slice. */
    seekable_file_t* file = nullptr;

    /** First byte of the range within the file. */
    std::size_t begin = 0;

    /** One past the last byte. */
    std::size_t end = 0;

    std::size_t size() const noexcept { return end - begin; }
    bool empty() const noexcept { return end == begin; }

    /** The @p length bytes at @p offset within this range; empty when they do not fit. */
    file_span_t subspan(std::size_t offset, std::size_t length) const noexcept {
        if (offset > size() || length > size() - offset) return {};
        return {file, begin + offset, begin + offset + length};
    }

    /** The same range, read-only. */
    file_view_t view() const noexcept { return {file, begin, end}; }
};

/**
 *  Writes @p source into the file bytes @p destination names, at their own offset.
 *
 *  @par Statuses:
 *  - @c config_mismatch_k The two ranges differ in size.
 *  - @c uninitialized_k A non-empty @p destination names no file.
 */
inline status_t copy_bytes(file_span_t destination, bytes_view_t source) noexcept {
    if (destination.size() != source.size()) return status_t::config_mismatch_k;
    if (source.empty()) return status_t::success_k;
    if (!destination.file) return status_t::uninitialized_k;
    return destination.file->write_at(destination.begin, source);
}

/**
 *  Reads the file bytes @p source names into @p destination.
 *
 *  @par Statuses:
 *  - @c malformed_k The two ranges differ in size.
 *  - @c uninitialized_k A non-empty @p source names no file.
 *  - @c io_k The read reaches the end of the file first.
 */
inline status_t copy_bytes(bytes_span_t destination, file_view_t source) noexcept {
    if (destination.size() != source.size()) return status_t::malformed_k;
    if (destination.empty()) return status_t::success_k;
    if (!source.file) return status_t::uninitialized_k;
    return source.file->read_at(source.begin, destination);
}

/**
 *  @brief A whole file mapped into the address space, so its bytes are a span and every reader
 *      and writer is the one that works on memory.
 *
 *  A read mapping opens what is there; a write mapping creates the file, sizes it once, and
 *  hands back zeroed bytes, since a freshly grown file reads as zeros - which is why nothing
 *  downstream sweeps the gaps between columns. WASI maps nothing, so there the mapping is a heap
 *  copy, filled once by @ref read and written back whole by @ref sync. A mapping is no durable
 *  replacement: a crash mid-write leaves a torn file, which only a part a manifest names may risk.
 */
class mapped_file_t {
  public:
    mapped_file_t() = default;
    mapped_file_t(mapped_file_t&& other) noexcept { swap(other); }
    mapped_file_t& operator=(mapped_file_t&& other) noexcept { return swap(other), *this; }
    mapped_file_t(mapped_file_t const&) = delete;
    mapped_file_t& operator=(mapped_file_t const&) = delete;
    ~mapped_file_t() noexcept { unmap_(); }

    void swap(mapped_file_t& other) noexcept {
        std::swap(data_, other.data_);
        std::swap(bytes_, other.bytes_);
        std::swap(access_, other.access_);
#if USEARCH_OS_WINDOWS_
        std::swap(file_handle_, other.file_handle_);
        std::swap(mapping_handle_, other.mapping_handle_);
#elif USEARCH_OS_WASI_
        std::swap(file_, other.file_);
#else
        std::swap(descriptor_, other.descriptor_);
        std::swap(byte_base_, other.byte_base_);
        std::swap(staged_bytes_, other.staged_bytes_);
#endif
    }

    /** Truthy exactly while a mapping is held. */
    explicit operator bool() const noexcept { return data_ != nullptr; }

    /** The mapped bytes, read-only. */
    bytes_view_t view() const noexcept { return {data_, bytes_}; }

    /** The mapped bytes for writing; empty on a read mapping. */
    bytes_span_t span() noexcept {
        return access_ != file_access_t::read_only_k ? bytes_span_t {data_, bytes_} : bytes_span_t {};
    }

    /** Bytes the mapping spans. */
    std::size_t size() const noexcept { return bytes_; }

    /** The mapped file opened again, read-only and unbuffered, beside the mapping; empty for a
     *  mapping short of its whole file, a filesystem refusing it, or a platform without
     *  @c /dev/fd, where the descriptor alone names the file. */
    seekable_file_t open_direct() const noexcept;

    /**
     *  Maps the whole of @p path read-only.
     *
     *  @par Statuses:
     *  - @c invalid_argument_k The path does not parse.
     *  - @c malformed_k The file is empty.
     */
    static expected<mapped_file_t> read(std::string_view path) noexcept;

    /**
     *  @brief Creates @p path at @p bytes and maps it writable; the bytes arrive zeroed.
     *
     *  On a disk the bytes stage in anonymous memory, and @ref sync writes them out direct.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k @p bytes is zero.
     *  - @c invalid_argument_k The path does not parse.
     *  - @c overflow_k A device ends before its base plus @p bytes.
     */
    static expected<mapped_file_t> write(std::string_view path, std::size_t bytes) noexcept;

    /** Makes a write mapping durable and releases it, surfacing what the destructor swallows. */
    status_t sync() noexcept;

  private:
    void unmap_() noexcept;

    byte_t* data_ = nullptr;
    std::size_t bytes_ = 0;
    file_access_t access_ = file_access_t::read_only_k;
#if USEARCH_OS_WINDOWS_
    HANDLE file_handle_ = INVALID_HANDLE_VALUE;
    HANDLE mapping_handle_ = nullptr;
#elif USEARCH_OS_WASI_
    seekable_file_t file_;
#else

    /** Writes the staged blocks out direct from the part's base, then flushes the disk. */
    status_t write_staged_() noexcept;

    native_file_t descriptor_ = invalid_native_file_k;

    /** Where the part's byte zero sits on the disk the staged bytes go out to. */
    u64_t byte_base_ = 0;

    /** Bytes staged in anonymous memory, whole logical blocks; zero for a shared file mapping. */
    std::size_t staged_bytes_ = 0;
#endif
};

#if USEARCH_OS_WINDOWS_

inline expected<mapped_file_t> mapped_file_t::read(std::string_view path) noexcept {
    mapped_file_t mapping;
    auto [file, open_status] = seekable_file_t::open(path, file_access_t::read_only_k);
    if (failed(open_status)) return {std::move(mapping), open_status};
    auto const [bytes, length_status] = file.length();
    if (failed(length_status)) return {std::move(mapping), length_status};
    if (!bytes) return {std::move(mapping), status_t::malformed_k};
    u64_t const byte_base = file.byte_base();
    mapping.file_handle_ = file.release();
    mapping.mapping_handle_ = ::CreateFileMappingA(mapping.file_handle_, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!mapping.mapping_handle_) return {std::move(mapping), status_t::io_k};
    void* base = ::MapViewOfFile(mapping.mapping_handle_, FILE_MAP_READ, static_cast<DWORD>(byte_base >> 32),
                                 static_cast<DWORD>(byte_base), bytes);
    if (!base) return {std::move(mapping), status_t::io_k};
    mapping.data_ = static_cast<byte_t*>(base);
    mapping.bytes_ = bytes;
    return {std::move(mapping), status_t::success_k};
}

inline expected<mapped_file_t> mapped_file_t::write(std::string_view path, std::size_t bytes) noexcept {
    mapped_file_t mapping;
    if (!bytes) return {std::move(mapping), status_t::config_mismatch_k};
    auto [file, status] = seekable_file_t::open(path, file_access_t::truncate_read_write_k);
    if (failed(status)) return {std::move(mapping), status};
    if (status_t const sized = file.truncate(bytes); failed(sized)) return {std::move(mapping), sized};
    u64_t const byte_base = file.byte_base();
    mapping.file_handle_ = file.release();
    mapping.mapping_handle_ = ::CreateFileMappingA(mapping.file_handle_, nullptr, PAGE_READWRITE, 0, 0, nullptr);
    if (!mapping.mapping_handle_) return {std::move(mapping), status_t::io_k};
    void* base = ::MapViewOfFile(mapping.mapping_handle_, FILE_MAP_WRITE, static_cast<DWORD>(byte_base >> 32),
                                 static_cast<DWORD>(byte_base), bytes);
    if (!base) return {std::move(mapping), status_t::io_k};
    mapping.data_ = static_cast<byte_t*>(base);
    mapping.bytes_ = bytes;
    mapping.access_ = file_access_t::truncate_read_write_k;
    return {std::move(mapping), status_t::success_k};
}

inline status_t mapped_file_t::sync() noexcept {
    if (!data_) return status_t::success_k;
    status_t flushed = status_t::success_k;
    // The view's flush hands its pages to the file system; the handle's flush makes them durable.
    if (access_ != file_access_t::read_only_k)
        flushed = ::FlushViewOfFile(data_, bytes_) ? flush_native_file(file_handle_) : status_t::io_k;
    unmap_();
    return flushed;
}

inline void mapped_file_t::unmap_() noexcept {
    if (data_) ::UnmapViewOfFile(data_);
    if (mapping_handle_) ::CloseHandle(mapping_handle_);
    if (file_handle_ != INVALID_HANDLE_VALUE) ::CloseHandle(file_handle_);
    data_ = nullptr, bytes_ = 0, access_ = file_access_t::read_only_k, mapping_handle_ = nullptr,
    file_handle_ = INVALID_HANDLE_VALUE;
}

#elif USEARCH_OS_WASI_

inline expected<mapped_file_t> mapped_file_t::read(std::string_view path) noexcept {
    mapped_file_t mapping;
    auto [file, open_status] = seekable_file_t::open(path, file_access_t::read_only_k);
    if (failed(open_status)) return {std::move(mapping), open_status};
    auto const [bytes, length_status] = file.length();
    if (failed(length_status)) return {std::move(mapping), length_status};
    if (!bytes) return {std::move(mapping), status_t::malformed_k};
    byte_t* base = std::allocator<byte_t> {}.allocate(bytes);
    if (!base) return {std::move(mapping), status_t::bad_alloc_k};
    if (status_t const filled = file.read_at(0, bytes_span_t {base, bytes}); failed(filled)) {
        std::allocator<byte_t> {}.deallocate(base, bytes);
        return {std::move(mapping), filled};
    }
    mapping.data_ = base;
    mapping.bytes_ = bytes;
    return {std::move(mapping), status_t::success_k};
}

inline expected<mapped_file_t> mapped_file_t::write(std::string_view path, std::size_t bytes) noexcept {
    mapped_file_t mapping;
    if (!bytes) return {std::move(mapping), status_t::config_mismatch_k};
    auto [file, status] = seekable_file_t::open(path, file_access_t::truncate_read_write_k);
    if (failed(status)) return {std::move(mapping), status};
    if (status_t const sized = file.truncate(bytes); failed(sized)) return {std::move(mapping), sized};
    byte_t* base = std::allocator<byte_t> {}.allocate(bytes);
    if (!base) return {std::move(mapping), status_t::bad_alloc_k};
    std::memset(base, 0, bytes);
    mapping.file_ = std::move(file);
    mapping.data_ = base;
    mapping.bytes_ = bytes;
    mapping.access_ = file_access_t::truncate_read_write_k;
    return {std::move(mapping), status_t::success_k};
}

inline status_t mapped_file_t::sync() noexcept {
    if (!data_) return status_t::success_k;
    status_t flushed = status_t::success_k;
    if (access_ != file_access_t::read_only_k) flushed = file_.write_at(0, bytes_view_t {data_, bytes_});
    if (access_ != file_access_t::read_only_k && succeeded(flushed)) flushed = file_.flush();
    unmap_();
    return flushed;
}

inline void mapped_file_t::unmap_() noexcept {
    if (data_) std::allocator<byte_t> {}.deallocate(data_, bytes_);
    [[maybe_unused]] status_t const closed = file_.close();
    data_ = nullptr, bytes_ = 0, access_ = file_access_t::read_only_k;
}

#else

inline expected<mapped_file_t> mapped_file_t::read(std::string_view path) noexcept {
    mapped_file_t mapping;
    auto [file, open_status] = seekable_file_t::open(path, file_access_t::read_only_k);
    if (failed(open_status)) return {std::move(mapping), open_status};
    auto const [bytes, length_status] = file.length();
    if (failed(length_status)) return {std::move(mapping), length_status};
    if (!bytes) return {std::move(mapping), status_t::malformed_k};
    off_t const byte_base = static_cast<off_t>(file.byte_base());
    mapping.descriptor_ = file.release();
    void* base = ::mmap(nullptr, bytes, PROT_READ, MAP_SHARED, mapping.descriptor_, byte_base);
    if (base == MAP_FAILED) return {std::move(mapping), status_t::io_k};
    mapping.data_ = static_cast<byte_t*>(base);
    mapping.bytes_ = bytes;
    return {std::move(mapping), status_t::success_k};
}

inline expected<mapped_file_t> mapped_file_t::write(std::string_view path, std::size_t bytes) noexcept {
    mapped_file_t mapping;
    if (!bytes) return {std::move(mapping), status_t::config_mismatch_k};
    auto [file, status] = seekable_file_t::open(path, file_access_t::truncate_read_write_k);
    if (failed(status)) return {std::move(mapping), status};
    if (status_t const sized = file.truncate(bytes); failed(sized)) return {std::move(mapping), sized};
    mapping.byte_base_ = file.byte_base();
    mapping.descriptor_ = file.release();
    // A disk has no holes, so a shared mapping would read every page before its first write.
    std::size_t const block_bytes = direct_write_block_bytes_(mapping.descriptor_);
    if (block_bytes && mapping.byte_base_ % block_bytes == 0)
        mapping.staged_bytes_ = round_up_to_multiple<std::size_t>(bytes, block_bytes);
    void* base = mapping.staged_bytes_ ? ::mmap(nullptr, mapping.staged_bytes_, PROT_READ | PROT_WRITE,
                                                MAP_PRIVATE | MAP_ANONYMOUS, -1, 0)
                                       : ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, mapping.descriptor_,
                                                static_cast<off_t>(mapping.byte_base_));
    if (base == MAP_FAILED) return {std::move(mapping), status_t::io_k};
    mapping.data_ = static_cast<byte_t*>(base);
    mapping.bytes_ = bytes;
    mapping.access_ = file_access_t::truncate_read_write_k;
    return {std::move(mapping), status_t::success_k};
}

inline status_t mapped_file_t::write_staged_() noexcept {
    // The tail past the part's last byte up to its block boundary goes out as zeros.
    i64_t const moved = write_native_file(descriptor_, native_file_event_t {}, bytes_view_t {data_, staged_bytes_},
                                          byte_base_);
    if (moved != static_cast<i64_t>(staged_bytes_)) return status_t::io_k;
    return flush_native_file(descriptor_);
}

inline status_t mapped_file_t::sync() noexcept {
    if (!data_) return status_t::success_k;
    status_t flushed = status_t::success_k;
    // `msync` hands the pages to the file system; Darwin's stay in the drive's cache until flushed.
    if (access_ != file_access_t::read_only_k && staged_bytes_) flushed = write_staged_();
    else if (access_ != file_access_t::read_only_k)
        flushed = ::msync(data_, bytes_, MS_SYNC) == 0 ? flush_native_file(descriptor_) : status_t::io_k;
    unmap_();
    return flushed;
}

inline seekable_file_t mapped_file_t::open_direct() const noexcept {
    // Reopening through the descriptor's own name gives the reader flags of its own.
    char path[32] = "/dev/fd/";
    *std::to_chars(path + 8, path + sizeof(path) - 1, descriptor_).ptr = '\0';
    seekable_file_t file;
    file.descriptor_ = open_native_file(path, file_access_t::read_only_k, file_buffering_t::unbuffered_k);
    if (!file) return file;
    // A mapping shorter than its file starts past byte zero, which offsets into it would miss.
    auto const [bytes, capacity_status] = native_file_capacity(file.descriptor_);
    if (failed(capacity_status) || bytes != bytes_) return seekable_file_t {};
    return file;
}

inline void mapped_file_t::unmap_() noexcept {
    if (data_) ::munmap(data_, staged_bytes_ ? staged_bytes_ : bytes_);
    if (descriptor_ != invalid_native_file_k) [[maybe_unused]]
        status_t const closed = close_native_file(descriptor_);
    data_ = nullptr, bytes_ = 0, access_ = file_access_t::read_only_k, descriptor_ = invalid_native_file_k,
    byte_base_ = 0, staged_bytes_ = 0;
}

#endif

#pragma endregion Slices and Mappings

#pragma region Durable Files

#if USEARCH_OS_WINDOWS_

/** Moves @p source over @p target in one step, replacing it, and returns once it is on disk. */
inline status_t rename_file(char const* source, char const* target) noexcept {
    return ::MoveFileExA(source, target, MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) ? status_t::success_k
                                                                                             : status_t::io_k;
}

/** Makes the entries of the directory at @p path durable, as a rename into it needs. */
inline status_t sync_directory(char const* path) noexcept {
    // Only a handle opened for backup semantics names a directory, and only a writable one flushes.
    native_file_t const directory = ::CreateFileA(path, GENERIC_WRITE,
                                                  FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                                                  OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS, nullptr);
    if (directory == invalid_native_file_k) return status_t::io_k;
    status_t const flushed = flush_native_file(directory);
    status_t const closed = close_native_file(directory);
    return failed(flushed) ? flushed : closed;
}

#else

/** Moves @p source over @p target in one step, replacing it. */
inline status_t rename_file(char const* source, char const* target) noexcept {
    return std::rename(source, target) == 0 ? status_t::success_k : status_t::io_k;
}

/** Makes the entries of the directory at @p path durable, as a rename into it needs. */
inline status_t sync_directory(char const* path) noexcept {
    native_file_t const directory = ::open(path, O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (directory == invalid_native_file_k) return status_t::io_k;
    // Darwin's `fsync` stops at the drive's cache; ext4 commits an entry only on a full `fsync`.
    status_t const flushed = USEARCH_OS_APPLE_         ? flush_native_file(directory)
                             : ::fsync(directory) == 0 ? status_t::success_k
                                                       : status_t::io_k;
    status_t const closed = close_native_file(directory);
    return failed(flushed) ? flushed : closed;
}

#endif // USEARCH_OS_WINDOWS_

/** Writes @p bytes into the fresh file @p staged, then moves it over @p target and makes the move
 *  durable in @p directory, removing @p staged on any failure before the move. */
inline status_t replace_file_durably_(char const* target, char const* staged, char const* directory,
                                      bytes_view_t bytes) noexcept {
    if (!names_regular_file_or_nothing_(target)) return status_t::invalid_argument_k;
    native_file_t const file = open_native_file(staged, file_access_t::truncate_read_write_k,
                                                file_buffering_t::buffered_k);
    if (file == invalid_native_file_k) return status_t::io_k;
    i64_t const written = write_native_file(file, native_file_event_t {}, bytes, 0);
    status_t replaced = written == static_cast<i64_t>(bytes.size()) ? status_t::success_k : status_t::io_k;
    if (succeeded(replaced)) replaced = flush_native_file(file);
    status_t const closed = close_native_file(file);
    if (succeeded(replaced)) replaced = closed;
    if (succeeded(replaced)) replaced = rename_file(staged, target);
    if (failed(replaced)) {
        [[maybe_unused]] int const removed = std::remove(staged);
        return replaced;
    }
    return sync_directory(directory);
}

/**
 *  @brief Replaces the file at @p path with @p bytes, so a crash at any point leaves either the old
 *      file or the new one whole, never a mix of the two.
 *
 *  Writes a sibling @c .tmp file, makes it durable and closes it, renames it over @p path, and
 *  makes the rename durable in the directory; a failure before the rename removes the sibling, and
 *  one after it leaves the new file in place. The three names the OS demands terminated are one
 *  transient allocation through @p allocator.
 *
 *  @par Statuses:
 *  - @c invalid_argument_k @p path is empty, or names a directory, a device or anything but a
 *    regular file.
 *  - @c bad_alloc_k The names could not be terminated.
 *  - @c io_k A write, a flush, the rename or the directory's flush failed.
 */
template <typename allocator_type_ = std::allocator<byte_t>>
status_t replace_file_durably(std::string_view path, bytes_view_t bytes, allocator_type_ allocator = {}) noexcept {
    constexpr std::string_view suffix = ".tmp";
    constexpr std::string_view separators = USEARCH_OS_WINDOWS_ ? "/\\" : "/";
    if (path.empty()) return status_t::invalid_argument_k;
    std::size_t const separator = path.find_last_of(separators);
    std::string_view const parent = separator == std::string_view::npos
                                        ? std::string_view {"."}
                                        : path.substr(0, max_of<std::size_t>(separator, 1));
    std::size_t const length = path.size();
    std::size_t const capacity = length + 1 + length + suffix.size() + 1 + parent.size() + 1;
    using chars_allocator_t = typename std::allocator_traits<allocator_type_>::template rebind_alloc<char>;
    chars_allocator_t chars {allocator};
    char* const target = chars.allocate(capacity);
    if (!target) return status_t::bad_alloc_k;
    char* const staged = target + length + 1;
    char* const directory = staged + length + suffix.size() + 1;
    target[path.copy(target, length)] = '\0';
    staged[path.copy(staged, length) + suffix.copy(staged + length, suffix.size())] = '\0';
    directory[parent.copy(directory, parent.size())] = '\0';
    status_t const replaced = replace_file_durably_(target, staged, directory, bytes);
    chars.deallocate(target, capacity);
    return replaced;
}

#pragma endregion Durable Files

} // namespace unum::usearch::utilities
