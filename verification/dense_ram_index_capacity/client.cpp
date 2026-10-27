/**
 *  @file verification/dense_ram_index_capacity/client.cpp
 *  @author Ash Vardanian
 *  @date September 9, 2026
 *  @brief GenMC client for the dense index's capacity protocol in
 *      `include/usearch/dense/ram_index.hpp`.
 *
 *  `claim_capacity` electing one grower, `release_capacity` releasing rows onto a non-negative
 *  counter only, and a claim that fits, spelled line for line over the ForkUnion reference the
 *  index binds. The header itself runs under GenMC, but its first chunk is a thousand rows of
 *  sentinel stores, more events than the execution graph can hold, so the counters stand alone
 *  here. The same scenario is @c growth.pml, which also runs it under far memory.
 *
 *  - Every claim's rows lie inside the chunks the claimer sees.
 *
 *  @verify pass sc,rc11,imm growth
 *  @verify fail rc11,imm growth_return_order_relaxed: relaxed, the directory read in the fitting
 *      claim races it
 *  @verify pass sc growth_return_order_relaxed
 */
#include <cstddef> // `std::size_t`, `std::ptrdiff_t` - the signed capacity word

#include <atomic> // `std::memory_order`

#include <genmc.hpp> // `spawn`, `join`, `verify` - the threads and the assertion GenMC intercepts

#include <forkunion/atomics.hpp> // `fu::standard_atomic_ref` - the reference the index binds

#include "usearch/utilities/types.hpp" // `u32_t` - the claim dispenser

namespace fu = ashvardanian::forkunion;
namespace us = unum::usearch;
using us::utilities::u32_t;
template <typename value_type_>
using atomic_ref = fu::standard_atomic_ref<value_type_>;

/** The header's choices; each mutant entry changes one. */
struct knobs_t {
    std::memory_order return_order = std::memory_order_release;
};

constexpr knobs_t header_k {};
constexpr knobs_t return_order_relaxed_k {.return_order = std::memory_order_relaxed};

constexpr std::size_t rows_per_chunk_k = 2;

/** The signed counter whose crossing elects one grower. */
std::ptrdiff_t remaining_capacity = 0;
/** The take: the first row of every claim. */
u32_t last_allocated_id = 0;
/** Plain, as in the index: the release ordering alone publishes it. */
std::size_t chunks_count = 0;

template <knobs_t const& knobs_>
u32_t claim_capacity(std::size_t count) noexcept {
    while (true) {
        std::ptrdiff_t const old_remaining =
            atomic_ref<std::ptrdiff_t>(remaining_capacity).fetch_sub(count, std::memory_order_acquire);
        std::ptrdiff_t const new_remaining = old_remaining - static_cast<std::ptrdiff_t>(count);
        if (old_remaining >= 0 && new_remaining < 0) {
            // Elected: the one grower until its release lands.
            chunks_count += 1;
            u32_t const first = atomic_ref<u32_t>(last_allocated_id).fetch_add(count, std::memory_order_acq_rel);
            verify(first + count <= chunks_count * rows_per_chunk_k);
            atomic_ref<std::ptrdiff_t>(remaining_capacity).add(rows_per_chunk_k, knobs_.return_order);
            return first;
        }
        if (new_remaining >= 0) {
            u32_t const first = atomic_ref<u32_t>(last_allocated_id).fetch_add(count, std::memory_order_acquire);
            verify(first + count <= chunks_count * rows_per_chunk_k);
            return first;
        }
        // A deficit someone else is filling: release the subtract and retry.
        atomic_ref<std::ptrdiff_t>(remaining_capacity).fetch_add(count, std::memory_order_release);
    }
}

/** Returns rows onto a non-negative counter only, so a grower inside is never lifted across zero. */
template <knobs_t const& knobs_>
void release_capacity(std::size_t count) noexcept {
    atomic_ref<u32_t>(last_allocated_id).sub(static_cast<u32_t>(count), std::memory_order_release);
    std::ptrdiff_t observed = atomic_ref<std::ptrdiff_t>(remaining_capacity).load(std::memory_order_acquire);
    while (observed < 0 || !atomic_ref<std::ptrdiff_t>(remaining_capacity)
                                .compare_exchange_weak(observed, observed + static_cast<std::ptrdiff_t>(count),
                                                       knobs_.return_order, std::memory_order_acquire))
        if (observed < 0) observed = atomic_ref<std::ptrdiff_t>(remaining_capacity).load(std::memory_order_acquire);
}

template <knobs_t const& knobs_>
void* grow(void*) noexcept {
    claim_capacity<knobs_>(2);
    release_capacity<knobs_>(2);
    return nullptr;
}

template <knobs_t const& knobs_>
void* claim(void*) noexcept {
    claim_capacity<knobs_>(1);
    return nullptr;
}

template <knobs_t const& knobs_>
int grow_and_claim() noexcept {
    thread_t const grower = spawn(grow<knobs_>);
    thread_t const claimer = spawn(claim<knobs_>);
    join(grower);
    join(claimer);
    return 0;
}

extern "C" int growth() { return grow_and_claim<header_k>(); }
extern "C" int growth_return_order_relaxed() { return grow_and_claim<return_order_relaxed_k>(); }
