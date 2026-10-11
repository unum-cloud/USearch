/**
 *  @file include/usearch/utilities/ram_sorts.hpp
 *  @author Ash Vardanian
 *  @date September 27, 2026
 *  @brief Orderings that never allocate: sorting networks for the smallest batches, an in-place
 *      radix grouping for the rest, and a heap selection of the first few, so no hot path calls a
 *      library sort.
 */
#pragma once

#include <cstddef> // `std::size_t`, `std::byte`, `std::to_integer`

#include <array>       // `std::array` - the bucket tables
#include <functional>  // `std::less`
#include <span>        // `std::span`, `std::as_bytes`
#include <type_traits> // `std::is_trivially_copyable_v`
#include <utility>     // `std::swap`

#include "usearch/utilities/types.hpp" // `u32_t`

namespace unum {
namespace usearch {

/**
 *  @brief Sorts up to four @p values ascending by @p comparator_type_ through fully-unrolled
 *      sorting networks - no loops, no recursion, no library sort; the four-element network
 *      is the optimal five compare-exchanges.
 *
 *  Exchanges fire only on strict inversions, so equal elements keep their original order -
 *  the stability the sorted-batch @c sorted_buffer::insert relies on for reproducible ties.
 */
template <typename element_type_, std::size_t extent_, typename comparator_type_ = std::less<void>>
inline void network_sort(std::span<element_type_, extent_> values, comparator_type_ comparator = {}) noexcept {
    auto exchange = [&](std::size_t left, std::size_t right) noexcept {
        if (comparator(values[right], values[left])) std::swap(values[left], values[right]);
    };
    switch (values.size()) {
    case 4: exchange(0, 1), exchange(2, 3), exchange(0, 2), exchange(1, 3), exchange(1, 2); return;
    case 3: exchange(0, 1), exchange(0, 2), exchange(1, 2); return;
    case 2: exchange(0, 1); return;
    default: return; // Zero or one element is sorted by definition.
    }
}

/**
 *  @brief Groups @p values in place so equal @p key members end up adjacent: an in-place MSD radix over the
 *      key's object bytes, needing two 256-entry tables per level on the stack and no other memory.
 *
 *  Equal keys land in no particular order among themselves, and the groups in no particular key order. At
 *  most @c u32_t values, since the bucket tables count in @c u32_t.
 */
template <typename value_type_, typename key_type_>
void radix_group(std::span<value_type_> values, key_type_ value_type_::* key, std::size_t position = 0) noexcept {
    static_assert(std::is_trivially_copyable_v<key_type_>, "Keys group by their object bytes");
    if (values.size() < 2 || position == sizeof(key_type_)) return;
    auto digit = [&](value_type_ const& value) noexcept {
        return std::to_integer<std::size_t>(std::as_bytes(std::span {&(value.*key), 1})[position]);
    };
    std::array<u32_t, 256> next {}, ends {};
    for (value_type_ const& value : values) ++ends[digit(value)];
    for (u32_t bucket = 0, start = 0; bucket != 256; ++bucket) {
        next[bucket] = start;
        start += ends[bucket];
        ends[bucket] = start;
    }
    // Each swap drops one value into its bucket for good, so the cycles finish in one sweep.
    for (std::size_t bucket = 0; bucket != 256; ++bucket)
        while (next[bucket] != ends[bucket]) {
            std::size_t const home = digit(values[next[bucket]]);
            if (home == bucket) ++next[bucket];
            else std::swap(values[next[bucket]], values[next[home]++]);
        }
    for (std::size_t bucket = 0, begin = 0; bucket != 256; begin = ends[bucket++])
        radix_group(values.subspan(begin, ends[bucket] - begin), key, position + 1);
}

/** Moves the first @p count of @p values under @p comparator to the front, ascending, in place -
 *  @c std::partial_sort re-spelled: a heap of the front keeps the worst kept on top, so a later
 *  value costs one compare unless it displaces that worst. */
template <typename value_type_, typename comparator_type_ = std::less<void>>
void partial_sort(std::span<value_type_> values, std::size_t count, comparator_type_ comparator = {}) noexcept {
    if (!count) return;
    auto sift_down = [&](std::size_t root, std::size_t size) noexcept {
        for (std::size_t child = 2 * root + 1; child < size; root = child, child = 2 * root + 1) {
            if (child + 1 < size && comparator(values[child], values[child + 1])) ++child;
            if (!comparator(values[root], values[child])) return;
            std::swap(values[root], values[child]);
        }
    };
    for (std::size_t parent = count / 2; parent-- != 0;) sift_down(parent, count);
    for (std::size_t candidate = count; candidate < values.size(); ++candidate)
        if (comparator(values[candidate], values[0])) {
            std::swap(values[0], values[candidate]);
            sift_down(0, count);
        }
    for (std::size_t size = count; size > 1;) {
        std::swap(values[0], values[--size]);
        sift_down(0, size);
    }
}

} // namespace usearch
} // namespace unum
