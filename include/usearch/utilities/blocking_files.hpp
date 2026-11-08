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
 */
#pragma once

#include <cerrno>  // `errno`, `EINTR`
#include <cstddef> // `std::byte`
#include <cstring> // `std::memcpy`, `std::memset`

#include <charconv>     // `std::from_chars`, for the keys of a file address
#include <memory>       // `std::allocator`, `std::allocator_traits`
#include <span>         // `std::span`
#include <string_view>  // `std::string_view`
#include <system_error> // `std::errc`
#include <utility>      // `std::exchange`

#include "usearch/utilities/types.hpp"

#if USEARCH_OS_WINDOWS_ && !defined(NOMINMAX)
#define NOMINMAX
#endif
#if USEARCH_OS_WINDOWS_
#include <windows.h> // `CreateFileA`, `CreateFileMappingA`, `MapViewOfFile`
#else
#include <fcntl.h>    // `open`
#include <sys/stat.h> // `fstat`, `S_ISBLK`
#include <unistd.h>   // `pread`, `pwrite`, `ftruncate`, `close`
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

#pragma region Descriptors

/** How @ref unum::usearch::utilities::seekable_file_t::open opens a file: reading an existing one,
 *  or creating a fresh one. */
enum class file_mode_t { read_k, write_k };

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
#if USEARCH_OS_WINDOWS_

    /** What the platform calls an open file. */
    using descriptor_t = HANDLE;

    /** The value a closed file carries. */
    inline static descriptor_t const closed_k = INVALID_HANDLE_VALUE;
#else

    /** What the platform calls an open file. */
    using descriptor_t = int;

    /** The value a closed file carries. */
    inline static constexpr descriptor_t closed_k = -1;
#endif

    seekable_file_t() = default;
    seekable_file_t(seekable_file_t&& other) noexcept
        : descriptor_(std::exchange(other.descriptor_, closed_k)), byte_base_(std::exchange(other.byte_base_, 0u)),
          byte_limit_(std::exchange(other.byte_limit_, 0u)) {}
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
    explicit operator bool() const noexcept { return descriptor_ != closed_k; }

    /**
     *  @brief Opens @p path, a file or a device in any form @ref file_address_t lists, every
     *      offset counting from its base.
     *
     *  The NUL terminator the OS demands is one transient allocation through @p allocator. A write
     *  past byte zero opens an existing medium, and a write to a disk opens it exclusively, so a
     *  mounted filesystem or another writer refuses it.
     *
     *  @par Statuses:
     *  - @c invalid_argument_k The path does not parse.
     */
    template <typename allocator_type_ = std::allocator<byte_t>>
    static expected<seekable_file_t> open(std::string_view path, file_mode_t mode,
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
        file.descriptor_ = open_(terminated, mode, parsed.byte_base);
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

    /** Flushes every write so far to the device, so a later write can be ordered after them. */
    status_t sync() noexcept;

    /** Closes, surfacing the failure the destructor would swallow. */
    status_t close() noexcept;

    /** Hands the descriptor to a caller that closes it from now on - what a mapping takes. */
    descriptor_t release() noexcept { return std::exchange(descriptor_, closed_k); }

    /** Where offset zero sits in the file, which a mapping starts at. */
    u64_t byte_base() const noexcept { return byte_base_; }

    /** The descriptor, still owned here, for a driver that registers the file. */
    descriptor_t descriptor() const noexcept { return descriptor_; }

  private:
    friend class mapped_file_t;

    /** Opens @p path under @p mode for a part starting @p byte_base bytes in, answering
     *  @c closed_k on failure. */
    static descriptor_t open_(char const* path, file_mode_t mode, u64_t byte_base) noexcept;

    /** The wrapped descriptor; @c closed_k marks the empty state a move-out leaves behind. */
    descriptor_t descriptor_ = closed_k;

    /** Where offset zero sits in the file. */
    u64_t byte_base_ = 0;

    /** Bytes past the base @ref truncate may size the file to, zero meaning to the medium's end. */
    u64_t byte_limit_ = 0;
};

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

inline seekable_file_t::descriptor_t seekable_file_t::open_(char const* path, file_mode_t mode,
                                                            u64_t byte_base) noexcept {
    if (mode == file_mode_t::read_k)
        return ::CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL,
                             nullptr);
    return ::CreateFileA(path, GENERIC_READ | GENERIC_WRITE, 0, nullptr, byte_base ? OPEN_EXISTING : CREATE_ALWAYS,
                         FILE_ATTRIBUTE_NORMAL, nullptr);
}

inline status_t seekable_file_t::truncate(std::size_t bytes) noexcept {
    if (byte_limit_ && bytes > byte_limit_) return status_t::overflow_k;
    LARGE_INTEGER distance;
    distance.QuadPart = static_cast<LONGLONG>(byte_base_ + bytes);
    if (!::SetFilePointerEx(descriptor_, distance, nullptr, FILE_BEGIN)) return status_t::io_k;
    return ::SetEndOfFile(descriptor_) ? status_t::success_k : status_t::io_k;
}

inline status_t seekable_file_t::write_at(std::size_t offset, bytes_view_t source) noexcept {
    offset += byte_base_;
    while (!source.empty()) {
        OVERLAPPED overlapped {};
        overlapped.Offset = static_cast<DWORD>(offset);
        overlapped.OffsetHigh = static_cast<DWORD>(offset >> 32);
        DWORD const asked = static_cast<DWORD>(min_of<std::size_t>(source.size(), 0x7FFFFFFFu));
        DWORD written = 0;
        if (!::WriteFile(descriptor_, source.data(), asked, &written, &overlapped) || !written) return status_t::io_k;
        source = source.subspan(written);
        offset += written;
    }
    return status_t::success_k;
}

inline status_t seekable_file_t::read_at(std::size_t offset, bytes_span_t destination) const noexcept {
    offset += byte_base_;
    while (!destination.empty()) {
        OVERLAPPED overlapped {};
        overlapped.Offset = static_cast<DWORD>(offset);
        overlapped.OffsetHigh = static_cast<DWORD>(offset >> 32);
        DWORD const asked = static_cast<DWORD>(min_of<std::size_t>(destination.size(), 0x7FFFFFFFu));
        DWORD read = 0;
        if (!::ReadFile(descriptor_, destination.data(), asked, &read, &overlapped) || !read) return status_t::io_k;
        destination = destination.subspan(read);
        offset += read;
    }
    return status_t::success_k;
}

inline expected<std::size_t> seekable_file_t::length() const noexcept {
    LARGE_INTEGER size;
    if (!::GetFileSizeEx(descriptor_, &size)) return {0, status_t::io_k};
    u64_t const bytes = static_cast<u64_t>(size.QuadPart);
    return {static_cast<std::size_t>(bytes - min_of(bytes, byte_base_)), status_t::success_k};
}

inline status_t seekable_file_t::sync() noexcept {
    return ::FlushFileBuffers(descriptor_) ? status_t::success_k : status_t::io_k;
}

inline status_t seekable_file_t::close() noexcept {
    if (descriptor_ == closed_k) return status_t::success_k;
    return ::CloseHandle(std::exchange(descriptor_, closed_k)) ? status_t::success_k : status_t::io_k;
}

#else

inline seekable_file_t::descriptor_t seekable_file_t::open_(char const* path, file_mode_t mode,
                                                            u64_t byte_base) noexcept {
    // A part past byte zero is written into a medium that exists, and the bytes before it stay.
    struct stat description;
    // An exclusive open of a disk fails while a filesystem or another writer holds it.
    int const exclusive =
        mode == file_mode_t::write_k && ::stat(path, &description) == 0 && S_ISBLK(description.st_mode) ? O_EXCL : 0;
    int const created = byte_base || exclusive ? exclusive : O_CREAT | O_TRUNC;
    int const flags = mode == file_mode_t::read_k ? O_RDONLY : (O_RDWR | created);
    return ::open(path, flags | O_CLOEXEC, 0644);
}

inline status_t seekable_file_t::truncate(std::size_t bytes) noexcept {
    if (byte_limit_ && bytes > byte_limit_) return status_t::overflow_k;
    struct stat description;
    if (::fstat(descriptor_, &description) != 0) return status_t::io_k;
    if (S_ISBLK(description.st_mode))
        return byte_base_ + bytes <= block_device_bytes_(descriptor_) ? status_t::success_k : status_t::overflow_k;
    return ::ftruncate(descriptor_, static_cast<off_t>(byte_base_ + bytes)) == 0 ? status_t::success_k : status_t::io_k;
}

inline status_t seekable_file_t::write_at(std::size_t offset, bytes_view_t source) noexcept {
    offset += byte_base_;
    while (!source.empty()) {
        ssize_t const written = ::pwrite(descriptor_, source.data(), source.size(), static_cast<off_t>(offset));
        if (written < 0 && errno == EINTR) continue;
        if (written <= 0) return status_t::io_k;
        source = source.subspan(static_cast<std::size_t>(written));
        offset += static_cast<std::size_t>(written);
    }
    return status_t::success_k;
}

inline status_t seekable_file_t::read_at(std::size_t offset, bytes_span_t destination) const noexcept {
    offset += byte_base_;
    while (!destination.empty()) {
        ssize_t const read = ::pread(descriptor_, destination.data(), destination.size(), static_cast<off_t>(offset));
        if (read < 0 && errno == EINTR) continue;
        if (read <= 0) return status_t::io_k;
        destination = destination.subspan(static_cast<std::size_t>(read));
        offset += static_cast<std::size_t>(read);
    }
    return status_t::success_k;
}

inline expected<std::size_t> seekable_file_t::length() const noexcept {
    struct stat description;
    if (::fstat(descriptor_, &description) != 0) return {0, status_t::io_k};
    u64_t const bytes = S_ISBLK(description.st_mode) ? block_device_bytes_(descriptor_)
                                                     : static_cast<u64_t>(description.st_size);
    return {static_cast<std::size_t>(bytes - min_of(bytes, byte_base_)), status_t::success_k};
}

inline status_t seekable_file_t::sync() noexcept {
    return ::fsync(descriptor_) == 0 ? status_t::success_k : status_t::io_k;
}

inline status_t seekable_file_t::close() noexcept {
    if (descriptor_ == closed_k) return status_t::success_k;
    return ::close(std::exchange(descriptor_, closed_k)) == 0 ? status_t::success_k : status_t::io_k;
}

#endif

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
 *  copy, filled once by @ref read and written back whole by @ref sync.
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
        std::swap(mode_, other.mode_);
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
        return mode_ == file_mode_t::write_k ? bytes_span_t {data_, bytes_} : bytes_span_t {};
    }

    /** Bytes the mapping spans. */
    std::size_t size() const noexcept { return bytes_; }

    /** The mapped regular file opened again, read-only and @c O_DIRECT, beside the mapping; empty
     *  for a device node, a mapping short of its whole file, a filesystem refusing it, or a
     *  platform without it. */
    seekable_file_t open_direct() const noexcept {
        seekable_file_t file;
#if USEARCH_OS_LINUX_
        char path[32] = "/proc/self/fd/";
        *std::to_chars(path + 14, path + sizeof(path) - 1, descriptor_).ptr = '\0';
        file.descriptor_ = ::open(path, O_RDONLY | O_DIRECT | O_CLOEXEC);
        struct stat description;
        // A mapping shorter than its file starts past byte zero, which offsets into it would miss.
        if (file && (::fstat(file.descriptor_, &description) != 0 || !S_ISREG(description.st_mode) ||
                     static_cast<std::size_t>(description.st_size) != bytes_))
            return seekable_file_t {};
#endif
        return file;
    }

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

    /** Flushes a write mapping and releases it, surfacing what the destructor swallows. */
    status_t sync() noexcept;

  private:
    void unmap_() noexcept;

    byte_t* data_ = nullptr;
    std::size_t bytes_ = 0;
    file_mode_t mode_ = file_mode_t::read_k;
#if USEARCH_OS_WINDOWS_
    HANDLE file_handle_ = INVALID_HANDLE_VALUE;
    HANDLE mapping_handle_ = nullptr;
#elif USEARCH_OS_WASI_
    seekable_file_t file_;
#else

    /** Writes the staged blocks out direct from the part's base, then flushes the disk. */
    status_t write_staged_() noexcept;

    int descriptor_ = -1;

    /** Where the part's byte zero sits on the disk the staged bytes go out to. */
    u64_t byte_base_ = 0;

    /** Bytes staged in anonymous memory, whole logical blocks; zero for a shared file mapping. */
    std::size_t staged_bytes_ = 0;
#endif
};

#if USEARCH_OS_WINDOWS_

inline expected<mapped_file_t> mapped_file_t::read(std::string_view path) noexcept {
    mapped_file_t mapping;
    auto [file, open_status] = seekable_file_t::open(path, file_mode_t::read_k);
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
    auto [file, status] = seekable_file_t::open(path, file_mode_t::write_k);
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
    mapping.mode_ = file_mode_t::write_k;
    return {std::move(mapping), status_t::success_k};
}

inline status_t mapped_file_t::sync() noexcept {
    if (!data_) return status_t::success_k;
    bool const flushed = mode_ == file_mode_t::read_k || ::FlushViewOfFile(data_, bytes_) != 0;
    unmap_();
    return flushed ? status_t::success_k : status_t::io_k;
}

inline void mapped_file_t::unmap_() noexcept {
    if (data_) ::UnmapViewOfFile(data_);
    if (mapping_handle_) ::CloseHandle(mapping_handle_);
    if (file_handle_ != INVALID_HANDLE_VALUE) ::CloseHandle(file_handle_);
    data_ = nullptr, bytes_ = 0, mode_ = file_mode_t::read_k, mapping_handle_ = nullptr,
    file_handle_ = INVALID_HANDLE_VALUE;
}

#elif USEARCH_OS_WASI_

inline expected<mapped_file_t> mapped_file_t::read(std::string_view path) noexcept {
    mapped_file_t mapping;
    auto [file, open_status] = seekable_file_t::open(path, file_mode_t::read_k);
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
    auto [file, status] = seekable_file_t::open(path, file_mode_t::write_k);
    if (failed(status)) return {std::move(mapping), status};
    if (status_t const sized = file.truncate(bytes); failed(sized)) return {std::move(mapping), sized};
    byte_t* base = std::allocator<byte_t> {}.allocate(bytes);
    if (!base) return {std::move(mapping), status_t::bad_alloc_k};
    std::memset(base, 0, bytes);
    mapping.file_ = std::move(file);
    mapping.data_ = base;
    mapping.bytes_ = bytes;
    mapping.mode_ = file_mode_t::write_k;
    return {std::move(mapping), status_t::success_k};
}

inline status_t mapped_file_t::sync() noexcept {
    if (!data_) return status_t::success_k;
    status_t const flushed = mode_ == file_mode_t::read_k ? status_t::success_k
                                                          : file_.write_at(0, bytes_view_t {data_, bytes_});
    unmap_();
    return flushed;
}

inline void mapped_file_t::unmap_() noexcept {
    if (data_) std::allocator<byte_t> {}.deallocate(data_, bytes_);
    [[maybe_unused]] status_t const closed = file_.close();
    data_ = nullptr, bytes_ = 0, mode_ = file_mode_t::read_k;
}

#else

inline expected<mapped_file_t> mapped_file_t::read(std::string_view path) noexcept {
    mapped_file_t mapping;
    auto [file, open_status] = seekable_file_t::open(path, file_mode_t::read_k);
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
    auto [file, status] = seekable_file_t::open(path, file_mode_t::write_k);
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
    mapping.mode_ = file_mode_t::write_k;
    return {std::move(mapping), status_t::success_k};
}

inline status_t mapped_file_t::write_staged_() noexcept {
    // The tail past the part's last byte up to its block boundary goes out as zeros.
    for (std::size_t written = 0; written != staged_bytes_;) {
        ssize_t const moved = ::pwrite(descriptor_, data_ + written, staged_bytes_ - written,
                                       static_cast<off_t>(byte_base_ + written));
        if (moved < 0 && errno == EINTR) continue;
        if (moved <= 0) return status_t::io_k;
        written += static_cast<std::size_t>(moved);
    }
    return ::fsync(descriptor_) == 0 ? status_t::success_k : status_t::io_k;
}

inline status_t mapped_file_t::sync() noexcept {
    if (!data_) return status_t::success_k;
    status_t flushed = status_t::success_k;
    if (mode_ == file_mode_t::write_k && staged_bytes_) flushed = write_staged_();
    else if (mode_ == file_mode_t::write_k && ::msync(data_, bytes_, MS_SYNC) != 0) flushed = status_t::io_k;
    unmap_();
    return flushed;
}

inline void mapped_file_t::unmap_() noexcept {
    if (data_) ::munmap(data_, staged_bytes_ ? staged_bytes_ : bytes_);
    if (descriptor_ >= 0) ::close(descriptor_);
    data_ = nullptr, bytes_ = 0, mode_ = file_mode_t::read_k, descriptor_ = -1, byte_base_ = 0, staged_bytes_ = 0;
}

#endif

#pragma endregion Slices and Mappings

} // namespace unum::usearch::utilities
