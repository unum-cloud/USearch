/**
 *  @file verification/genmc.hpp
 *  @author Ash Vardanian
 *  @date September 9, 2026
 *  @brief What a GenMC client takes from the verifier: threads, and the assertion it reports with
 *      the execution graph that broke it.
 *
 *  GenMC intercepts these by name. The platform's @c pthread_create it does not, and @c std::thread
 *  rides on that, so every client spawns through here. The clients compile against the platform's
 *  standard library rather than GenMC's freestanding replacement, which `check.sh` arranges by
 *  forwarding the header names GenMC shadows.
 */
#pragma once
#include <cstddef> // `size_t` - what `genmc_internal.h` uses without declaring

#include <genmc_internal.h> // `__VERIFIER_thread_create`, `__VERIFIER_thread_join`, `__VERIFIER_assert_fail`

/** A thread GenMC created and will join. */
using thread_t = __VERIFIER_thread_t;

/** Starts @p routine on @p argument in a thread GenMC intercepts. */
inline thread_t spawn(void* (*routine)(void*), void* argument = nullptr) noexcept {
    return __VERIFIER_thread_create(nullptr, routine, argument);
}

/** Waits for @p thread to return. */
inline void join(thread_t thread) noexcept { __VERIFIER_thread_join(thread); }

/** The assertion GenMC reports with the execution graph that broke it. */
#define verify(expression) ((expression) ? (void)0 : __VERIFIER_assert_fail(#expression, __FILE__, __LINE__))
