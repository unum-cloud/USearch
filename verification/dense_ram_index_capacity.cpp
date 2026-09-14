/**
 *  @brief GenMC client for the dense index's capacity protocol in `include/usearch/dense/ram_index.hpp`:
 *      `claim_capacity` electing one grower, `release_capacity` releasing rows onto a
 *      non-negative counter only, and a claim
 *      that fits, spelled line for line over the ForkUnion reference the index binds. The
 *      header itself runs under GenMC, but its first chunk is a thousand rows of sentinel
 *      stores, more events than the execution graph can hold, so the counters stand alone here.
 *  @author Ash Vardanian
 *  @file verification/dense_ram_index_capacity.cpp
 *  @date September 9, 2026
 *
 *  Thread 0 claims two rows, which grows the only chunk, and releases them; thread 1 claims
 *  one. Every claim's rows lie inside the chunks the claimer sees. The chunk directory is plain
 *  memory ordered only by the acquire on the subtract and the released store that returns the rows,
 *  and GenMC's race detector checks that chain: `-Dwithout_ordered_release` weakens that store, and
 *  the directory read in the fitting claim races the grower's write. The same scenario is
 *  `dense_ram_index_capacity.pml`, which also runs it under far memory.
 */
#include <cstddef> // `std::size_t`, `std::ptrdiff_t` - the signed capacity word

#include <atomic> // `std::memory_order`

#include <forkunion/atomics.hpp> // `fu::standard_atomic_ref` - the reference the index binds

#include "usearch/utilities/types.hpp" // `u32_t` - the claim dispenser

#include "genmc.hpp"

namespace fu = ashvardanian::forkunion;
namespace us = unum::usearch;
using us::u32_t;
template <typename value_type_>
using atomic_ref = fu::standard_atomic_ref<value_type_>;

constexpr std::size_t rows_per_chunk_k = 2;
/** @brief The ordering of the store that returns rows; the knob weakens it to relaxed. */
#ifdef without_ordered_release
constexpr std::memory_order release_order_k = std::memory_order_relaxed;
#else
constexpr std::memory_order release_order_k = std::memory_order_release;
#endif

/** @brief The signed counter whose crossing elects one grower. */
std::ptrdiff_t remaining_capacity = 0;
/** @brief The take: the first row of every claim. */
u32_t last_allocated_id = 0;
/** @brief Plain, as in the index: the release ordering alone publishes it. */
std::size_t chunks_count = 0;

u32_t claim_capacity(std::size_t count) noexcept {
    while (true) {
        std::ptrdiff_t const old_remaining =
            atomic_ref<std::ptrdiff_t>(remaining_capacity).fetch_sub(count, std::memory_order_acquire);
        std::ptrdiff_t const new_remaining = old_remaining - static_cast<std::ptrdiff_t>(count);
        if (old_remaining >= 0 && new_remaining < 0) {
            // Elected: the one grower until its release lands.
            chunks_count += 1;
            u32_t const first =
                atomic_ref<u32_t>(last_allocated_id).fetch_add(count, std::memory_order_acq_rel);
            verify(first + count <= chunks_count * rows_per_chunk_k);
            atomic_ref<std::ptrdiff_t>(remaining_capacity).add(rows_per_chunk_k, release_order_k);
            return first;
        }
        if (new_remaining >= 0) {
            u32_t const first =
                atomic_ref<u32_t>(last_allocated_id).fetch_add(count, std::memory_order_acquire);
            verify(first + count <= chunks_count * rows_per_chunk_k);
            return first;
        }
        // A deficit someone else is filling: release the subtract and retry.
        atomic_ref<std::ptrdiff_t>(remaining_capacity).fetch_add(count, std::memory_order_release);
    }
}

/** @brief Returns rows onto a non-negative counter only, so a grower inside is never lifted across zero. */
void release_capacity(std::size_t count) noexcept {
    atomic_ref<u32_t>(last_allocated_id).sub(static_cast<u32_t>(count), std::memory_order_release);
    std::ptrdiff_t observed = atomic_ref<std::ptrdiff_t>(remaining_capacity).load(std::memory_order_acquire);
    while (observed < 0 || !atomic_ref<std::ptrdiff_t>(remaining_capacity)
                                .compare_exchange_weak(observed, observed + static_cast<std::ptrdiff_t>(count),
                                                       release_order_k, std::memory_order_acquire))
        if (observed < 0) observed = atomic_ref<std::ptrdiff_t>(remaining_capacity).load(std::memory_order_acquire);
}

void* grow(void*) noexcept {
    claim_capacity(2);
    release_capacity(2);
    return nullptr;
}

void* claim(void*) noexcept {
    claim_capacity(1);
    return nullptr;
}

int main() {
    thread_t const grower = spawn(grow);
    thread_t const claimer = spawn(claim);
    join(grower);
    join(claimer);
    return 0;
}
