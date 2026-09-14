/**
 *  @brief Files as bytes, two ways: @ref seekable_file_t addressed by offset with sliceable
 *      proxies over its contents, and @ref mapped_file_t addressed as memory - so a reader and
 *      a writer both speak @c subspan and @ref copy_bytes wherever the bytes sit.
 *
 *  Every verb here blocks the calling thread, which is the whole difference from the NVMe
 *  engines: those queue their reads and return, where these park until the kernel answers.
 *  Bulk work - a whole checkpoint, a whole dataset - is what this is for, and it needs no
 *  @c io_uring, no I/O Rings and no @c O_DIRECT, so it builds and runs everywhere.
 *
 *  @author Ash Vardanian
 *  @file include/usearch/utilities/blocking_files.hpp
 *  @date September 4, 2026
 *
 *  The format this backs is memory-mappable by design, so reading is a mapping and every reader
 *  works on @c bytes_view_t. Writing has two shapes for one reason: a mapping dirties the page
 *  cache, which a terabyte checkpoint should not, so @ref file_span_t addresses a descriptor
 *  positionally and @ref copy_bytes puts the difference in an overload rather than at a call site.
 *  Slices carry their own absolute offset, which is what an aligned direct write will want.
 */
#pragma once

#include <cerrno>  // `errno`, `EINTR`
#include <cstddef> // `std::byte`
#include <cstring> // `std::memcpy`, `std::memset`

#include <memory>      // `std::allocator`, `std::allocator_traits`
#include <span>        // `std::span`
#include <string_view> // `std::string_view`
#include <utility>     // `std::exchange`

#if defined(_WIN32)
#if !defined(NOMINMAX)
#define NOMINMAX
#endif
#include <windows.h> // `CreateFileA`, `CreateFileMappingA`, `MapViewOfFile`
#else
#include <fcntl.h>    // `open`
#include <sys/stat.h> // `fstat`
#include <unistd.h>   // `pread`, `pwrite`, `ftruncate`, `close`
#if !defined(__wasi__)
#include <sys/mman.h> // `mmap`, `munmap`, `msync`
#endif
#endif

#include "usearch/utilities/types.hpp"

namespace unum {
namespace usearch {

#pragma region Descriptors

/** @brief How @ref seekable_file_t::open opens a file: reading an existing one, or creating a fresh one. */
enum class file_mode_t { read_k, write_k };

/** @brief An owned file descriptor with positional verbs only - no cursor, so any thread reads or
 *      writes any offset; every verb answers @c status_t. */
struct seekable_file_t {
#if defined(_WIN32)
    /** @brief What the platform calls an open file. */
    using descriptor_t = HANDLE;
    /** @brief The value a closed file carries. */
    inline static descriptor_t const closed_k = INVALID_HANDLE_VALUE;
#else
    /** @brief What the platform calls an open file. */
    using descriptor_t = int;
    /** @brief The value a closed file carries. */
    inline static constexpr descriptor_t closed_k = -1;
#endif

    seekable_file_t() = default;
    seekable_file_t(seekable_file_t&& other) noexcept : descriptor_(std::exchange(other.descriptor_, closed_k)) {}
    seekable_file_t& operator=(seekable_file_t&& other) noexcept {
        return std::swap(descriptor_, other.descriptor_), *this;
    }
    seekable_file_t(seekable_file_t const&) = delete;
    seekable_file_t& operator=(seekable_file_t const&) = delete;
    ~seekable_file_t() noexcept { [[maybe_unused]] status_t const closed = close(); }

    /** @brief Truthy exactly while a file is open. */
    explicit operator bool() const noexcept { return descriptor_ != closed_k; }

    /** @brief Opens @p path; the NUL terminator the OS demands is one transient allocation
     *      through @p allocator. */
    template <typename allocator_type_ = std::allocator<byte_t>>
    static expected<seekable_file_t> open(std::string_view path, file_mode_t mode,
                                          allocator_type_ allocator = {}) noexcept {
        using chars_allocator_t = typename std::allocator_traits<allocator_type_>::template rebind_alloc<char>;
        chars_allocator_t chars {allocator};
        char* terminated = chars.allocate(path.size() + 1);
        if (!terminated) return {seekable_file_t {}, status_t::bad_alloc_k};
        std::memcpy(terminated, path.data(), path.size());
        terminated[path.size()] = '\0';
        seekable_file_t file;
        file.descriptor_ = open_(terminated, mode);
        chars.deallocate(terminated, path.size() + 1);
        status_t const opened = file ? status_t::success_k : status_t::io_k;
        return {std::move(file), opened};
    }

    /** @brief Sets the file's length to @p bytes, so positional writes land in a pre-sized file. */
    status_t truncate(std::size_t bytes) noexcept;
    /** @brief Writes every byte of @p source at @p offset, looping over short writes. */
    status_t write_at(std::size_t offset, bytes_view_t source) noexcept;
    /** @brief Reads exactly @p destination.size() bytes at @p offset; a short read answers @c io_k. */
    status_t read_at(std::size_t offset, bytes_span_t destination) const noexcept;
    /** @brief Byte length of the file. */
    expected<std::size_t> length() const noexcept;
    /** @brief Closes, surfacing the failure the destructor would swallow. */
    status_t close() noexcept;
    /** @brief Hands the descriptor to a caller that closes it from now on - what a mapping takes. */
    descriptor_t release() noexcept { return std::exchange(descriptor_, closed_k); }

  private:
    /** @brief Opens @p path under @p mode, answering @c closed_k on failure. */
    static descriptor_t open_(char const* path, file_mode_t mode) noexcept;

    /** @brief The wrapped descriptor; @c closed_k marks the empty state a move-out leaves behind. */
    descriptor_t descriptor_ = closed_k;
};

#if defined(_WIN32)

inline seekable_file_t::descriptor_t seekable_file_t::open_(char const* path, file_mode_t mode) noexcept {
    if (mode == file_mode_t::read_k)
        return ::CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL,
                             nullptr);
    return ::CreateFileA(path, GENERIC_READ | GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
}

inline status_t seekable_file_t::truncate(std::size_t bytes) noexcept {
    LARGE_INTEGER distance;
    distance.QuadPart = static_cast<LONGLONG>(bytes);
    if (!::SetFilePointerEx(descriptor_, distance, nullptr, FILE_BEGIN)) return status_t::io_k;
    return ::SetEndOfFile(descriptor_) ? status_t::success_k : status_t::io_k;
}

inline status_t seekable_file_t::write_at(std::size_t offset, bytes_view_t source) noexcept {
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
    return {static_cast<std::size_t>(size.QuadPart), status_t::success_k};
}

inline status_t seekable_file_t::close() noexcept {
    if (descriptor_ == closed_k) return status_t::success_k;
    return ::CloseHandle(std::exchange(descriptor_, closed_k)) ? status_t::success_k : status_t::io_k;
}

#else

inline seekable_file_t::descriptor_t seekable_file_t::open_(char const* path, file_mode_t mode) noexcept {
    int const flags = mode == file_mode_t::read_k ? O_RDONLY : (O_RDWR | O_CREAT | O_TRUNC);
    return ::open(path, flags | O_CLOEXEC, 0644);
}

inline status_t seekable_file_t::truncate(std::size_t bytes) noexcept {
    return ::ftruncate(descriptor_, static_cast<off_t>(bytes)) == 0 ? status_t::success_k : status_t::io_k;
}

inline status_t seekable_file_t::write_at(std::size_t offset, bytes_view_t source) noexcept {
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
    return {static_cast<std::size_t>(description.st_size), status_t::success_k};
}

inline status_t seekable_file_t::close() noexcept {
    if (descriptor_ == closed_k) return status_t::success_k;
    return ::close(std::exchange(descriptor_, closed_k)) == 0 ? status_t::success_k : status_t::io_k;
}

#endif

#pragma endregion Descriptors

#pragma region Slices and Mappings

/** @brief A read-only run of a file's bytes: the descriptor and an absolute byte range, sliced
 *      like a span and read through @ref copy_bytes. */
struct file_view_t {
    /** @brief The file the range lives in; null for an empty slice. */
    seekable_file_t const* file = nullptr;
    /** @brief First byte of the range within the file. */
    std::size_t begin = 0;
    /** @brief One past the last byte. */
    std::size_t end = 0;

    std::size_t size() const noexcept { return end - begin; }
    bool empty() const noexcept { return end == begin; }
    /** @brief The @p length bytes at @p offset within this range; empty when they do not fit. */
    file_view_t subspan(std::size_t offset, std::size_t length) const noexcept {
        if (offset > size() || length > size() - offset) return {};
        return {file, begin + offset, begin + offset + length};
    }
};

/** @brief The writable twin of @ref file_view_t, written through @ref copy_bytes. */
struct file_span_t {
    /** @brief The file the range lives in; null for an empty slice. */
    seekable_file_t* file = nullptr;
    /** @brief First byte of the range within the file. */
    std::size_t begin = 0;
    /** @brief One past the last byte. */
    std::size_t end = 0;

    std::size_t size() const noexcept { return end - begin; }
    bool empty() const noexcept { return end == begin; }
    /** @brief The @p length bytes at @p offset within this range; empty when they do not fit. */
    file_span_t subspan(std::size_t offset, std::size_t length) const noexcept {
        if (offset > size() || length > size() - offset) return {};
        return {file, begin + offset, begin + offset + length};
    }
    /** @brief The same range, read-only. */
    file_view_t view() const noexcept { return {file, begin, end}; }
};

/** @brief Writes @p source into the file bytes @p destination names, at their own offset. */
inline status_t copy_bytes(file_span_t destination, bytes_view_t source) noexcept {
    if (destination.size() != source.size()) return status_t::config_mismatch_k;
    if (source.empty()) return status_t::success_k;
    if (!destination.file) return status_t::uninitialized_k;
    return destination.file->write_at(destination.begin, source);
}

/** @brief Reads the file bytes @p source names into @p destination. */
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
#if defined(_WIN32)
        std::swap(file_handle_, other.file_handle_);
        std::swap(mapping_handle_, other.mapping_handle_);
#elif defined(__wasi__)
        std::swap(file_, other.file_);
#else
        std::swap(descriptor_, other.descriptor_);
#endif
    }

    /** @brief Truthy exactly while a mapping is held. */
    explicit operator bool() const noexcept { return data_ != nullptr; }
    /** @brief The mapped bytes, read-only. */
    bytes_view_t view() const noexcept { return {data_, bytes_}; }
    /** @brief The mapped bytes for writing; empty on a read mapping. */
    bytes_span_t span() noexcept {
        return mode_ == file_mode_t::write_k ? bytes_span_t {data_, bytes_} : bytes_span_t {};
    }
    /** @brief Bytes the mapping spans. */
    std::size_t size() const noexcept { return bytes_; }

    /** @brief Maps the whole of @p path read-only. */
    static expected<mapped_file_t> read(std::string_view path) noexcept;
    /** @brief Creates @p path at @p bytes and maps it writable; the bytes arrive zeroed. */
    static expected<mapped_file_t> write(std::string_view path, std::size_t bytes) noexcept;
    /** @brief Flushes a write mapping and releases it, surfacing what the destructor swallows. */
    status_t sync() noexcept;

  private:
    void unmap_() noexcept;

    byte_t* data_ = nullptr;
    std::size_t bytes_ = 0;
    file_mode_t mode_ = file_mode_t::read_k;
#if defined(_WIN32)
    HANDLE file_handle_ = INVALID_HANDLE_VALUE;
    HANDLE mapping_handle_ = nullptr;
#elif defined(__wasi__)
    seekable_file_t file_;
#else
    int descriptor_ = -1;
#endif
};

#if defined(_WIN32)

inline expected<mapped_file_t> mapped_file_t::read(std::string_view path) noexcept {
    mapped_file_t mapping;
    auto [file, opened] = seekable_file_t::open(path, file_mode_t::read_k);
    if (failed(opened)) return {std::move(mapping), opened};
    auto const [bytes, measured] = file.length();
    if (failed(measured)) return {std::move(mapping), measured};
    if (!bytes) return {std::move(mapping), status_t::malformed_k};
    mapping.file_handle_ = file.release();
    mapping.mapping_handle_ = ::CreateFileMappingA(mapping.file_handle_, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!mapping.mapping_handle_) return {std::move(mapping), status_t::io_k};
    void* base = ::MapViewOfFile(mapping.mapping_handle_, FILE_MAP_READ, 0, 0, 0);
    if (!base) return {std::move(mapping), status_t::io_k};
    mapping.data_ = static_cast<byte_t*>(base);
    mapping.bytes_ = bytes;
    return {std::move(mapping), status_t::success_k};
}

inline expected<mapped_file_t> mapped_file_t::write(std::string_view path, std::size_t bytes) noexcept {
    mapped_file_t mapping;
    if (!bytes) return {std::move(mapping), status_t::config_mismatch_k};
    auto [file, opened] = seekable_file_t::open(path, file_mode_t::write_k);
    if (failed(opened)) return {std::move(mapping), opened};
    if (status_t const sized = file.truncate(bytes); failed(sized)) return {std::move(mapping), sized};
    mapping.file_handle_ = file.release();
    mapping.mapping_handle_ = ::CreateFileMappingA(mapping.file_handle_, nullptr, PAGE_READWRITE, 0, 0, nullptr);
    if (!mapping.mapping_handle_) return {std::move(mapping), status_t::io_k};
    void* base = ::MapViewOfFile(mapping.mapping_handle_, FILE_MAP_WRITE, 0, 0, 0);
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

#elif defined(__wasi__)

inline expected<mapped_file_t> mapped_file_t::read(std::string_view path) noexcept {
    mapped_file_t mapping;
    auto [file, opened] = seekable_file_t::open(path, file_mode_t::read_k);
    if (failed(opened)) return {std::move(mapping), opened};
    auto const [bytes, measured] = file.length();
    if (failed(measured)) return {std::move(mapping), measured};
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
    auto [file, opened] = seekable_file_t::open(path, file_mode_t::write_k);
    if (failed(opened)) return {std::move(mapping), opened};
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
    auto [file, opened] = seekable_file_t::open(path, file_mode_t::read_k);
    if (failed(opened)) return {std::move(mapping), opened};
    auto const [bytes, measured] = file.length();
    if (failed(measured)) return {std::move(mapping), measured};
    if (!bytes) return {std::move(mapping), status_t::malformed_k};
    mapping.descriptor_ = file.release();
    void* base = ::mmap(nullptr, bytes, PROT_READ, MAP_SHARED, mapping.descriptor_, 0);
    if (base == MAP_FAILED) return {std::move(mapping), status_t::io_k};
    mapping.data_ = static_cast<byte_t*>(base);
    mapping.bytes_ = bytes;
    return {std::move(mapping), status_t::success_k};
}

inline expected<mapped_file_t> mapped_file_t::write(std::string_view path, std::size_t bytes) noexcept {
    mapped_file_t mapping;
    if (!bytes) return {std::move(mapping), status_t::config_mismatch_k};
    auto [file, opened] = seekable_file_t::open(path, file_mode_t::write_k);
    if (failed(opened)) return {std::move(mapping), opened};
    if (status_t const sized = file.truncate(bytes); failed(sized)) return {std::move(mapping), sized};
    mapping.descriptor_ = file.release();
    void* base = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, mapping.descriptor_, 0);
    if (base == MAP_FAILED) return {std::move(mapping), status_t::io_k};
    mapping.data_ = static_cast<byte_t*>(base);
    mapping.bytes_ = bytes;
    mapping.mode_ = file_mode_t::write_k;
    return {std::move(mapping), status_t::success_k};
}

inline status_t mapped_file_t::sync() noexcept {
    if (!data_) return status_t::success_k;
    bool const flushed = mode_ == file_mode_t::read_k || ::msync(data_, bytes_, MS_SYNC) == 0;
    unmap_();
    return flushed ? status_t::success_k : status_t::io_k;
}

inline void mapped_file_t::unmap_() noexcept {
    if (data_) ::munmap(data_, bytes_);
    if (descriptor_ >= 0) ::close(descriptor_);
    data_ = nullptr, bytes_ = 0, mode_ = file_mode_t::read_k, descriptor_ = -1;
}

#endif

#pragma endregion Slices and Mappings

} // namespace usearch
} // namespace unum
