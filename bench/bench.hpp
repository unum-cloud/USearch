/**
 *  @file bench/bench.hpp
 *  @author Ash Vardanian
 *  @date July 17, 2026
 *  @brief Logging helpers shared by the benchmarking harnesses.
 *
 *  Everything here is presentation-only: progress that redraws one line on a terminal and prints deciles when
 *  piped, human-readable count formatting, enum pretty-printers, and process-level memory telemetry. No benchmark
 *  logic belongs in this header.
 */
#pragma once

#include <cstddef> // `std::size_t`
#include <cstdio>  // `std::fopen` for `/proc/self/status`
#include <cstdlib> // `std::strtoull`
#include <cstring> // `std::strncmp`

#include <chrono> // `std::chrono::steady_clock` for progress rates

#include <fmt/format.h>

#include <numkong/types.h>

#include "usearch/utilities/types.hpp"

namespace unum::usearch::bench {

/** A count or rate rendered as "1.23 G" / "45.6 M" / "789 k" for scannable tables. */
template <typename value_type_>
struct humanized {
    value_type_ value;
    explicit humanized(value_type_ value) noexcept : value(value) {}
};

} // namespace unum::usearch::bench

namespace us = unum::usearch;
namespace nk = ashvardanian::numkong;

/** Renders straight into the output, so a progress tick allocates nothing. */
template <typename value_type_>
struct fmt::formatter<unum::usearch::bench::humanized<value_type_>> {
    constexpr auto parse(format_parse_context& context) { return context.begin(); }
    auto format(us::bench::humanized<value_type_> wrapped, format_context& context) const {
        double const value = static_cast<double>(wrapped.value);
        if (value >= 1e9) return fmt::format_to(context.out(), "{:.2f} G", value / 1e9);
        if (value >= 1e6) return fmt::format_to(context.out(), "{:.2f} M", value / 1e6);
        if (value >= 1e3) return fmt::format_to(context.out(), "{:.2f} k", value / 1e3);
        return fmt::format_to(context.out(), "{:.0f}", value);
    }
};

namespace unum::usearch::bench {

namespace stdc = std::chrono;
using steady_clock_t = stdc::steady_clock;
using time_point_t = steady_clock_t::time_point;

/** A byte count in the gigabytes that reports print as "GB". */
constexpr double gigabytes(std::size_t bytes) noexcept { return double(bytes) / (1ull << 30); }

/**
 *  @brief Phase progress for every usage form: an in-place line with a bar, rate, and ETA
 *      on a TTY, or one plain line per completed decile when piped or logged - short and
 *      ANSI-free where scrollback is permanent.
 *
 *  Tick with the completed count from the driving loop; @c finish clears the TTY line so
 *  the phase report takes its place.
 */
class progress_bar_t {
  public:
    progress_bar_t(char const* label, std::size_t total, bool terminal) noexcept
        : label_(label), total_(total), terminal_(terminal) {}

    void tick(std::size_t completed) noexcept {
        if (terminal_) render_bar_(completed);
        else render_decile_(completed);
    }
    void finish() noexcept {
        if (terminal_) fmt::print("\r\033[K");
    }

  private:
    inline static constexpr std::size_t bar_cells_k = 24;

    char const* label_;
    std::size_t total_;
    bool terminal_;
    time_point_t started_ {steady_clock_t::now()};
    time_point_t rendered_ {};
    std::size_t reported_deciles_ = 0;

    double rate_per_second_(std::size_t completed) const noexcept {
        double const seconds = stdc::duration<double>(steady_clock_t::now() - started_).count();
        return seconds > 0 ? static_cast<double>(completed) / seconds : 0;
    }

    void render_bar_(std::size_t completed) noexcept {
        auto const now = steady_clock_t::now();
        bool const final = completed == total_;
        if (!final && now - rendered_ < stdc::milliseconds(100)) return; // The work outruns the terminal.
        rendered_ = now;
        std::size_t const filled_cells = total_ ? bar_cells_k * completed / total_ : bar_cells_k;
        double const rate = rate_per_second_(completed);
        double const eta_seconds = rate > 0 ? static_cast<double>(total_ - completed) / rate : 0;
        fmt::print("\r\033[K  {} {:█<{}}{:░<{}} {} / {} · {}/s · eta {:.0f} s", label_, "", filled_cells, "",
                   bar_cells_k - filled_cells, humanized(completed), humanized(total_), humanized(rate), eta_seconds);
        std::fflush(stdout);
    }

    void render_decile_(std::size_t completed) noexcept {
        std::size_t const deciles = total_ ? 10 * completed / total_ : 10;
        if (completed != total_ && deciles == reported_deciles_) return;
        reported_deciles_ = deciles;
        fmt::println("  {}: {} / {} · {}/s", label_, humanized(completed), humanized(total_),
                     humanized(rate_per_second_(completed)));
        // A piped stdout is block-buffered: without the flush a decile line can lag its
        // phase by minutes in `tee`-style logs, which exist precisely to watch progress.
        std::fflush(stdout);
    }
};

/** Names the benchmarked dtype subset; NumKong exposes no C-level name table yet. */
inline char const* dtype_name(nk_dtype_t dtype) noexcept {
    switch (dtype) {
    // Sorted by decreasing dynamic range, grouped floats, signed integers, unsigned.
    case nk_f32_k: return "f32";
    case nk_bf16_k: return "bf16";
    case nk_f16_k: return "f16";
    case nk_e5m2_k: return "e5m2";
    case nk_e4m3_k: return "e4m3";
    case nk_e3m2_k: return "e3m2";
    case nk_e2m3_k: return "e2m3";
    case nk_i8_k: return "i8";
    case nk_i4_k: return "i4";
    case nk_u32_k: return "u32";
    case nk_u8_k: return "u8";
    case nk_u4_k: return "u4";
    case nk_u1_k: return "u1";
    default: return "unknown";
    }
}

/** Peak resident set size from @c /proc/self/status in bytes, or zero when unknown. */
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

} // namespace unum::usearch::bench
