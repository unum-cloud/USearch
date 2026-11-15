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

#include "usearch/utilities/types.hpp" // `u32_t`, `round_hubs`

namespace unum::usearch::utilities {

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

/** Replaces the farthest of the max-heap @p heap under @p further with @p entry. */
template <typename entry_type_, typename further_type_>
void replace_farthest(std::span<entry_type_> heap, entry_type_ entry, further_type_ const& further) noexcept {
    std::size_t parent = 0;
    for (std::size_t child = 1; child < heap.size(); parent = child, child = 2 * parent + 1) {
        if (child + 1 < heap.size() && further(heap[child + 1], heap[child])) ++child;
        if (!further(heap[child], entry)) break;
        heap[parent] = heap[child];
    }
    heap[parent] = entry;
}

/** Slots @ref split_wide_neighborhoods may write over @p entries entries drawing at most
 *  @p hubs_per_draw hubs a gather: one start per entry and one closing them, past one split's
 *  starts and cursors. */
constexpr std::size_t split_room(std::size_t entries, std::size_t hubs_per_draw) noexcept {
    return entries + 2 * hubs_per_draw + 3;
}

/** What an order by neighborhood leaves: the neighborhoods, whose starts lead the offsets, and the
 *  widest one's entries. */
struct neighborhood_order_t {
    std::size_t neighborhoods = 0;
    std::size_t widest = 0;
};

/** Merges the repeated starts of the ascending @p offsets in place, which hubs no entry chose
 *  leave. */
template <typename node_id_type_>
neighborhood_order_t merge_repeated_starts(std::span<node_id_type_> offsets) noexcept {
    std::size_t kept = 1, widest = 0;
    for (std::size_t boundary = 1; boundary != offsets.size(); ++boundary) {
        if (offsets[boundary] == offsets[kept - 1]) continue;
        widest = max_of<std::size_t>(widest, offsets[boundary] - offsets[kept - 1]);
        offsets[kept++] = offsets[boundary];
    }
    return {kept - 1, widest};
}

/**
 *  @brief Sorts the entries by the neighborhood @p entry_neighborhoods gives each, among @p hubs,
 *      into @p entries, and writes where each neighborhood starts into @p offsets.
 *
 *  Each neighborhood's entries ascend, and one start past the last closes them. @p offsets holds
 *  at least @p hubs + 2 slots: the counts land two ahead, so after the prefix sums the placement
 *  advances each neighborhood's start to its end, which is the next one's start.
 */
template <typename node_id_type_>
neighborhood_order_t order_by_neighborhood(std::span<u16_t const> entry_neighborhoods, std::size_t hubs,
                                           std::span<node_id_type_> entries,
                                           std::span<node_id_type_> offsets) noexcept {
    std::size_t const visible = entry_neighborhoods.size();
    for (std::size_t slot = 0; slot != hubs + 2; ++slot) offsets[slot] = 0;
    for (std::size_t entry = 0; entry != visible; ++entry) ++offsets[entry_neighborhoods[entry] + 2u];
    for (std::size_t hub = 2; hub < hubs + 2; ++hub) offsets[hub] += offsets[hub - 1];
    for (std::size_t entry = 0; entry != visible; ++entry)
        entries[offsets[entry_neighborhoods[entry] + 1u]++] = static_cast<node_id_type_>(entry);
    return merge_repeated_starts(offsets.first(hubs + 1));
}

/**
 *  @brief Splits every one of the @p count neighborhoods of @p offsets wider than @p capacity in
 *      place, so no neighborhood stays over capacity.
 *
 *  A wide neighborhood's entries sort by the neighborhood their last gather gave each, the new
 *  ones ascending by ID; one that gather left whole, its every entry nearest to one hub as
 *  duplicates are, is cut into pieces by ID instead. New starts go into @p offsets past its
 *  @p count + 1, which must hold as many slots as @ref split_room counts; then the starts sort and
 *  their repeats merge.
 */
template <typename node_id_type_>
neighborhood_order_t split_wide_neighborhoods(std::span<u16_t const> entry_neighborhoods,
                                              std::span<node_id_type_> entries, std::span<node_id_type_> offsets,
                                              std::size_t count, std::size_t capacity,
                                              std::size_t hubs_per_draw) noexcept {
    std::size_t starts_count = count + 1;
    for (std::size_t neighborhood = 0; neighborhood != count; ++neighborhood) {
        std::size_t const width = offsets[neighborhood + 1] - offsets[neighborhood];
        std::size_t const hubs = round_hubs(width, hubs_per_draw, capacity);
        if (width <= capacity) continue;
        node_id_type_ const first = offsets[neighborhood];
        std::span<node_id_type_> const members = entries.subspan(first, width);
        // Past the starts sit each new neighborhood's start, then its cursor, both from zero.
        std::size_t const appended = starts_count;
        node_id_type_* const starts = offsets.data() + appended;
        node_id_type_* const cursors = starts + hubs + 1;
        for (std::size_t slot = 0; slot != 2 * hubs + 1; ++slot) starts[slot] = 0;
        for (node_id_type_ const member : members) ++starts[entry_neighborhoods[member] + 1u];
        node_id_type_ largest = 0;
        for (std::size_t hub = 1; hub <= hubs; ++hub) largest = max_of(largest, starts[hub]);
        if (largest == width) {
            std::size_t const pieces = divide_round_up(width, capacity);
            std::size_t const piece = divide_round_up(width, pieces);
            starts_count = appended;
            for (std::size_t cut = piece; cut < width; cut += piece)
                offsets[starts_count++] = static_cast<node_id_type_>(first + cut);
            continue;
        }
        for (std::size_t hub = 1; hub <= hubs; ++hub) starts[hub] += starts[hub - 1];
        for (std::size_t hub = 0; hub != hubs; ++hub) cursors[hub] = starts[hub];
        // Each swap moves one entry into its new neighborhood for good.
        for (std::size_t hub = 0; hub != hubs; ++hub)
            while (cursors[hub] < starts[hub + 1]) {
                std::size_t const into = entry_neighborhoods[members[cursors[hub]]];
                if (into == hub) ++cursors[hub];
                else std::swap(members[cursors[hub]], members[cursors[into]++]);
            }
        for (std::size_t hub = 0; hub != hubs; ++hub)
            partial_sort(members.subspan(starts[hub], starts[hub + 1] - starts[hub]), starts[hub + 1] - starts[hub]);
        // Each start moves down past the empty neighborhoods onto its own slot or an earlier one.
        std::size_t written = appended;
        for (std::size_t hub = 1, last = 0; hub != hubs; ++hub) {
            node_id_type_ const start = starts[hub];
            if (start == last || start == width) continue;
            offsets[written++] = static_cast<node_id_type_>(first + start);
            last = start;
        }
        starts_count = written;
    }
    std::span<node_id_type_> const kept = offsets.first(starts_count);
    partial_sort(kept, kept.size());
    return merge_repeated_starts(kept);
}

} // namespace unum::usearch::utilities
