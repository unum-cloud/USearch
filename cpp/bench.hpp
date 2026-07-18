/**
 *  @brief Logging & styling helpers shared by the benchmarking harnesses.
 *
 *  @author Ash Vardanian
 *  @file cpp/bench.hpp
 *  @date July 17, 2026
 *
 *  Everything here is presentation-only: TTY-aware {fmt} styles that degrade to plain text
 *  when piped, human-readable count formatting, enum pretty-printers, and process-level
 *  memory telemetry. No benchmark logic belongs in this header.
 */
#pragma once

#include <cstddef> // `std::size_t`
#include <cstdio>  // `std::fopen` for `/proc/self/status`, `stdout` for the TTY probe
#include <cstdlib> // `std::strtoull`
#include <cstring> // `std::strncmp`

#include <string> // `std::string`

#if !defined(_WIN32)
#include <unistd.h> // `isatty`
#endif

#include <fmt/color.h>
#include <fmt/format.h>

#include <numkong/types.h>

namespace unum::usearch {

inline bool stdout_is_terminal() noexcept {
#if defined(_WIN32)
    return false;
#else
    return isatty(fileno(stdout)) != 0;
#endif
}

/** @brief Bold for section headers on a TTY; empty when piped, so logs stay ANSI-free. */
inline fmt::text_style header_style() noexcept {
    return stdout_is_terminal() ? fmt::text_style(fmt::emphasis::bold) : fmt::text_style();
}

/** @brief Green when a judgment value looks healthy, orange otherwise; empty when piped. */
inline fmt::text_style verdict_style(bool looks_healthy) noexcept {
    if (!stdout_is_terminal()) return {};
    return fg(looks_healthy ? fmt::color::green : fmt::color::orange);
}

/** @brief Renders large counts as "1.23 G" / "45.6 M" / "789 k" for scannable tables. */
inline std::string humanize(double value) {
    if (value >= 1e9) return fmt::format("{:.2f} G", value / 1e9);
    if (value >= 1e6) return fmt::format("{:.2f} M", value / 1e6);
    if (value >= 1e3) return fmt::format("{:.2f} k", value / 1e3);
    return fmt::format("{:.0f}", value);
}

inline char const* dtype_name(nk_dtype_t dtype) noexcept {
    switch (dtype) {
    case nk_f32_k: return "f32";
    case nk_bf16_k: return "bf16";
    case nk_i8_k: return "i8";
    case nk_u8_k: return "u8";
    case nk_i4_k: return "i4";
    case nk_u32_k: return "u32";
    default: return "unknown";
    }
}

inline char const* kernel_name(nk_kernel_kind_t kernel) noexcept {
    switch (kernel) {
    case nk_kernel_dot_k: return "dot";
    case nk_kernel_angular_k: return "angular";
    case nk_kernel_euclidean_k: return "euclidean";
    default: return "unknown";
    }
}

/** @brief Peak resident set size from `/proc/self/status` in bytes, or zero when unknown. */
inline std::size_t peak_resident_bytes() noexcept {
    std::FILE* status = std::fopen("/proc/self/status", "r");
    if (!status) return 0;
    char line[256];
    std::size_t resident_bytes = 0;
    while (std::fgets(line, sizeof(line), status))
        if (std::strncmp(line, "VmHWM:", 6) == 0) {
            resident_bytes = std::strtoull(line + 6, nullptr, 10) * 1024;
            break;
        }
    std::fclose(status);
    return resident_bytes;
}

} // namespace unum::usearch
