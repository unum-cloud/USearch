/**
 *  @brief Cache-control & prefetch hints for 1-to-N graph-traversal distance compute.
 *  @author Ash Vardanian
 *  @file include/usearch/ram_hints.hpp
 *  @date April 22, 2026
 */
#pragma once

namespace unum {
namespace usearch {

/** @brief No-op fallback used as the primary-template base for every role. */
struct memory_hint_t {
    inline void operator()(void const*) const noexcept {}
};

#if defined(__CLDEMOTE__)

/**
 *  @brief Demote one cache line from the core's private caches (L1/L2) to the last shared level.
 *
 *  Advisory: the CPU may drop the hint under contention. The line is not invalidated, just moved
 *  down the hierarchy, so subsequent reads from any core hit L3 instead of having to snoop a
 *  remote private cache.
 *
 *  Respected on Intel Tremont, Ice Lake-SP, Alder Lake P-cores, Sapphire Rapids and newer; AMD
 *  Zen 4 (Genoa) and Zen 5 (Turin). Off-target the opcode raises @c #UD - the @c __CLDEMOTE__
 *  guard restricts emission to compile targets that admit it.
 *
 *  @see https://www.felixcloutier.com/x86/cldemote
 */
struct demote_candidate_cldemote_t {
    inline void operator()(void const* p) const noexcept { asm volatile("cldemote (%0)" : : "r"(p) : "memory"); }
};

#endif

#if defined(__x86_64__)

/**
 *  @brief Fill one cache line into L1 (and L2, L3 by inclusion) with strong temporal-locality hint.
 *
 *  Universally honoured by every x86 core since the Pentium III (1999); part of baseline SSE so
 *  always legal on @c __x86_64__ (SSE2 is mandated by the ABI). Older in-order Atom cores
 *  treated it as a no-op; modern E-cores (Gracemont, Crestmont) execute it. Like every prefetch
 *  it is advisory and may be silently dropped under load.
 *
 *  @see https://www.felixcloutier.com/x86/prefetchh
 */
struct prefetch_query_sse_t {
    inline void operator()(void const* p) const noexcept { asm volatile("prefetcht0 (%0)" : : "r"(p) : "memory"); }
};

/**
 *  @brief Fill one cache line targeting L2/L3 with low-temporal-locality hint, skipping L1.
 *
 *  Advisory; distinction from @c T0 / @c T1 / @c NTA is implementation-defined and uneven. Intel
 *  Haswell through Skylake honoured @c T2 as "land in L2+L3, not L1"; on Ice Lake and later
 *  Intel's Optimization Manual notes that @c T1 and @c T2 may be treated identically. AMD Zen 2
 *  through Zen 5 distinguishes @c T2 cleanly - the line lands in L3 only. Older in-order Atom
 *  and VIA cores often treated it as a no-op. For pure stream-through candidate traffic on AMD,
 *  @c PREFETCHNTA bypasses L1/L2 more aggressively than @c T2.
 *
 *  @see https://www.felixcloutier.com/x86/prefetchh
 */
struct prefetch_candidate_sse_t {
    inline void operator()(void const* p) const noexcept { asm volatile("prefetcht2 (%0)" : : "r"(p) : "memory"); }
};

#endif

#if defined(__aarch64__)

/**
 *  @brief Prefetch one cache line into L1 with normal-LRU policy ("keep" - high temporal locality).
 *
 *  The @c PRFM instruction is part of the ARMv8-A baseline, so always legal on @c __aarch64__.
 *  Honoured by every shipping AArch64 core (Cortex-A, Neoverse, Apple M-series), though some
 *  in-order Cortex-A53/A55 cores reduce its effect. Like every prefetch it is advisory.
 *
 *  @see https://www.scs.stanford.edu/~zyedidia/arm64/prfm_imm.html
 */
struct prefetch_query_prfm_t {
    inline void operator()(void const* p) const noexcept { asm volatile("prfm pldl1keep, [%0]" : : "r"(p)); }
};

/**
 *  @brief Prefetch one cache line into L3 with streaming policy ("strm" - low temporal locality).
 *
 *  The @c strm policy marks the line for early eviction so it does not occupy normal-LRU slots,
 *  and targets the L3 level so L1 and L2 stay free for hotter data. Honoured by Neoverse
 *  N1/V1/N2/V2 and Apple M-series; older Cortex-A cores treat @c strm identically to @c keep.
 *
 *  @see https://www.scs.stanford.edu/~zyedidia/arm64/prfm_imm.html
 */
struct prefetch_candidate_prfm_t {
    inline void operator()(void const* p) const noexcept { asm volatile("prfm pldl3strm, [%0]" : : "r"(p)); }
};

#endif

} // namespace usearch
} // namespace unum
