/**
 *  @file       persistent_index_helpers.hpp
 *  @author     Mikhail Chichvarin
 *  @brief      Helper utilities for `persistent_index_gt`: CRC32, RAII `FILE*`,
 *              WAL/manifest record headers and op codes.
 */
#ifndef UNUM_USEARCH_PERSISTENT_INDEX_HELPERS_HPP
#define UNUM_USEARCH_PERSISTENT_INDEX_HELPERS_HPP

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>

namespace unum {
namespace usearch {

/// @brief  Bytewise CRC32 (IEEE polynomial). Records are short; no table.
inline std::uint32_t crc32_ieee(void const* data, std::size_t bytes) noexcept {
    std::uint8_t const* p = static_cast<std::uint8_t const*>(data);
    std::uint32_t crc = 0xFFFFFFFFu;
    for (std::size_t i = 0; i != bytes; ++i) {
        crc ^= p[i];
        for (int k = 0; k != 8; ++k)
            crc = (crc >> 1) ^ ((crc & 1u) ? 0xEDB88320u : 0u);
    }
    return ~crc;
}

enum persistent_op_t : std::uint8_t {
    persistent_op_add_k = 1,
    persistent_op_remove_k = 2,
};

/// @brief  RAII wrapper over a C `FILE*`: closes on scope exit, drops every
///         `std::fclose(file); return error_t(...)` paired cleanup.
struct file_closer_t {
    void operator()(std::FILE* f) const noexcept {
        if (f)
            std::fclose(f);
    }
};
using file_t = std::unique_ptr<std::FILE, file_closer_t>;

/// @brief  Self-describing header at the start of every WAL file.
struct persistent_wal_header_t {
    char magic[4]; // "uwal"
    std::uint32_t format_version;
    std::uint64_t dimensions;
    std::uint32_t scalar_kind;
    std::uint32_t metric_kind;
};
static constexpr char const* persistent_wal_magic_k = "uwal";
static constexpr std::uint32_t persistent_wal_version_k = 1;

/// @brief  Atomic pointer to the current durable generation.
struct persistent_manifest_t {
    char magic[4]; // "umft"
    std::uint32_t format_version;
    std::uint64_t generation;
};
static constexpr char const* persistent_manifest_magic_k = "umft";

} // namespace usearch
} // namespace unum

#endif // UNUM_USEARCH_PERSISTENT_INDEX_HELPERS_HPP
