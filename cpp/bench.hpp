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

#include <cmath>   // `std::log2` for ranking gains
#include <cstddef> // `std::size_t`
#include <cstdio>  // `std::fopen` for `/proc/self/status`, `stdout` for the TTY probe
#include <cstdlib> // `std::strtoull`
#include <cstring> // `std::strncmp`

#include <span>   // `std::span`
#include <string> // `std::string`

#if !defined(_WIN32)
#include <unistd.h> // `isatty`
#endif

#include <fmt/color.h>
#include <fmt/format.h>

#include <numkong/types.h>

#include "usearch/types.hpp"

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

/** @brief Names the benchmarked dtype subset; NumKong exposes no C-level name table yet. */
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

/** @brief Names the benchmarked kernel subset; NumKong exposes no C-level name table yet. */
inline char const* kernel_name(nk_kernel_kind_t kernel) noexcept {
    switch (kernel) {
    case nk_kernel_dot_k: return "dot";
    case nk_kernel_angular_k: return "angular";
    case nk_kernel_euclidean_k: return "euclidean";
    default: return "unknown";
    }
}

/**
 *  @brief Precomputes the position-discount tables shared by every query's judgment, so
 *      per-query normalization is one lookup instead of a chain of @c log2 calls.
 *
 *  @param[out] discounts Per-position pay-offs: @c discounts[i] rewards a hit returned at position i.
 *  @param[out] ideal Prefix sums of @p discounts: @c ideal[i] is the gain of a perfect ranking
 *      of length i + 1.
 */
inline void fill_discounts(std::span<double> discounts, std::span<double> ideal) noexcept {
    double running = 0;
    for (std::size_t position = 0; position != discounts.size(); ++position) {
        discounts[position] = 1.0 / std::log2(double(position) + 2.0);
        running += discounts[position];
        if (position < ideal.size()) ideal[position] = running;
    }
}

/** @brief The verdicts of one query's judgment, produced together by @c judge_ranking. */
struct ranking_verdict_t {
    /** @brief 1 when the true nearest neighbor appears among the returned keys. Averaged
     *      over queries this is the production Recall@K - the probability of finding @b the
     *      nearest neighbor within the top K - monotone in K, so Recall@1 never exceeds
     *      Recall@10. */
    std::size_t found_nearest = 0;
    /** @brief The intersection size between the returned and expected sets - "overlap",
     *      the set-coverage metric big-ANN leaderboards report. Stricter than Recall@K and
     *      not monotone against it. */
    std::size_t overlap = 0;
    /** @brief Unnormalized discounted cumulative gain: every returned key present in the
     *      expected set earns its position's discount, so early hits weigh more. */
    double gain = 0;
};

/**
 *  @brief All ranking verdicts for one query in one sweep: the expected keys are sorted once,
 *      then every returned position pays one binary search - O(k·log k) where the naive
 *      nested probes go O(k²), which matters from top-100 recalls up.
 *
 *  @param[in] returned Keys the index produced, best first.
 *  @param[in] expected Ground-truth keys, ascending by true distance - @c expected[0] is the
 *      true nearest neighbor.
 *  @param[inout] scratch Clobbered workspace for sorting the expected keys; its length caps
 *      how many of them are judged.
 *  @param[in] discounts Position pay-offs from @c fill_discounts.
 */
template <typename key_type_>
ranking_verdict_t judge_ranking(std::span<key_type_ const> returned, std::span<key_type_ const> expected,
                                std::span<key_type_> scratch, std::span<double const> discounts) noexcept {
    std::size_t const expected_count = expected.size() < scratch.size() ? expected.size() : scratch.size();
    for (std::size_t sorted = 0; sorted != expected_count; ++sorted) {
        key_type_ const key = expected[sorted];
        key_type_* const slot = lower_bound(scratch.subspan(0, sorted), key);
        std::memmove(slot + 1, slot, (scratch.data() + sorted - slot) * sizeof(key_type_));
        *slot = key;
    }

    ranking_verdict_t verdict;
    if (!expected_count) return verdict;
    key_type_ const expected_nearest = expected.front();
    std::span<key_type_ const> const sorted {scratch.data(), expected_count};
    for (std::size_t position = 0; position != returned.size(); ++position) {
        key_type_ const key = returned[position];
        verdict.found_nearest |= static_cast<std::size_t>(key == expected_nearest);
        key_type_ const* const slot = lower_bound(sorted, key);
        if (slot != sorted.data() + sorted.size() && *slot == key) {
            verdict.overlap += 1;
            if (position < discounts.size()) verdict.gain += discounts[position];
        }
    }
    return verdict;
}

/** @brief Peak resident set size from @c /proc/self/status in bytes, or zero when unknown. */
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
