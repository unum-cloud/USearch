/**
 *  @file include/usearch/utilities/sanitizers.hpp
 *  @author Ash Vardanian
 *  @date August 27, 2026
 *  @brief Sanitizer ABI shims: mutex annotations for the thread sanitizer & contiguous- container
 *      annotations for the address sanitizer, compiling to no-ops elsewhere.
 *
 *  The thread half teaches TSan the home-grown locks: every verb brackets its atomic with
 *  pre/post annotations, so race reports name the held mutexes and lock-order inversions
 *  surface. Inside a bracket TSan ignores the verb's own atomics, so a happens-before edge
 *  that pairs a bare acquire load with an unlock's release - the certified probes' whole
 *  mechanism - would silently vanish; @c annotate_release ahead of the unlock brackets and
 *  @c annotate_acquire beside the probe restore that edge through the documented
 *  @c __tsan_acquire / @c __tsan_release contract. No create/destroy annotations: an
 *  unannounced mutex passes its flags at @c pre_lock, and skipping creation keeps the lock
 *  types trivially destructible in every configuration.
 *
 *  The address half re-poisons a flat container's capacity slack, so a read past its size
 *  reports as a container overflow instead of sailing through owned memory. The two halves
 *  never coexist - both compilers refuse to combine the sanitizers - and the container verb
 *  must stay inert under TSan alone: @c libtsan does not export it.
 */
#pragma once

/*  GCC defines `__SANITIZE_THREAD__` / `__SANITIZE_ADDRESS__`; Clang answers `__has_feature`.
 *  The feature test stays nested: GCC 13 has no `__has_feature`, and a one-line
 *  `defined(__has_feature) && __has_feature(...)` is a hard preprocessor error there - the
 *  right operand is parsed even though `&&` short-circuits. */
#if defined(__SANITIZE_THREAD__)
#define USEARCH_SANITIZE_THREAD 1
#elif defined(__has_feature)
#if __has_feature(thread_sanitizer)
#define USEARCH_SANITIZE_THREAD 1
#endif
#endif
#if !defined(USEARCH_SANITIZE_THREAD)
#define USEARCH_SANITIZE_THREAD 0
#endif

#if defined(__SANITIZE_ADDRESS__)
#define USEARCH_SANITIZE_ADDRESS 1
#elif defined(__has_feature)
#if __has_feature(address_sanitizer)
#define USEARCH_SANITIZE_ADDRESS 1
#endif
#endif
#if !defined(USEARCH_SANITIZE_ADDRESS)
#define USEARCH_SANITIZE_ADDRESS 0
#endif

#if USEARCH_SANITIZE_THREAD
#include <sanitizer/tsan_interface.h> // `__tsan_mutex_pre_lock` & the rest of the mutex ABI
#elif USEARCH_SANITIZE_ADDRESS
#include <sanitizer/common_interface_defs.h> // `__sanitizer_annotate_contiguous_container`
#endif

namespace unum {
namespace usearch {

#if USEARCH_SANITIZE_THREAD

inline constexpr bool thread_sanitizer_k = true;
inline constexpr bool address_sanitizer_k = false;

/** Opens the bracket of a blocking @c lock; the verb's own atomics are ignored
 *  until the matching @c annotate_lock_acquired. */
inline void annotate_lock_started(void const volatile* mutex_address) noexcept {
    __tsan_mutex_pre_lock(const_cast<void*>(mutex_address), 0u);
}

/** Closes the @c lock bracket: the calling thread now holds the mutex. */
inline void annotate_lock_acquired(void const volatile* mutex_address) noexcept {
    __tsan_mutex_post_lock(const_cast<void*>(mutex_address), 0u, 0);
}

/** Opens the bracket of a @c try_lock; closes through @c annotate_try_lock_acquired
 *  or @c annotate_try_lock_refused - the flags must match this opening. */
inline void annotate_try_lock_started(void const volatile* mutex_address) noexcept {
    __tsan_mutex_pre_lock(const_cast<void*>(mutex_address), __tsan_mutex_try_lock);
}

/** Closes a @c try_lock bracket as a successful acquisition. */
inline void annotate_try_lock_acquired(void const volatile* mutex_address) noexcept {
    __tsan_mutex_post_lock(const_cast<void*>(mutex_address), __tsan_mutex_try_lock, 0);
}

/** Closes a @c try_lock bracket as a refusal - nothing is held. */
inline void annotate_try_lock_refused(void const volatile* mutex_address) noexcept {
    __tsan_mutex_post_lock(const_cast<void*>(mutex_address), __tsan_mutex_try_lock | __tsan_mutex_try_lock_failed, 0);
}

/** Opens an @c unlock bracket; the release into the mutex happens at the opening. */
inline void annotate_unlock_started(void const volatile* mutex_address) noexcept {
    __tsan_mutex_pre_unlock(const_cast<void*>(mutex_address), 0u);
}

/** Closes an @c unlock bracket. */
inline void annotate_unlock_finished(void const volatile* mutex_address) noexcept {
    __tsan_mutex_post_unlock(const_cast<void*>(mutex_address), 0u);
}

/** The shared-side twin of @c annotate_lock_started, carrying the read-lock flag. */
inline void annotate_lock_shared_started(void const volatile* mutex_address) noexcept {
    __tsan_mutex_pre_lock(const_cast<void*>(mutex_address), __tsan_mutex_read_lock);
}

/** The shared-side twin of @c annotate_lock_acquired. */
inline void annotate_lock_shared_acquired(void const volatile* mutex_address) noexcept {
    __tsan_mutex_post_lock(const_cast<void*>(mutex_address), __tsan_mutex_read_lock, 0);
}

/** The shared-side twin of @c annotate_unlock_started. */
inline void annotate_unlock_shared_started(void const volatile* mutex_address) noexcept {
    __tsan_mutex_pre_unlock(const_cast<void*>(mutex_address), __tsan_mutex_read_lock);
}

/** The shared-side twin of @c annotate_unlock_finished. */
inline void annotate_unlock_shared_finished(void const volatile* mutex_address) noexcept {
    __tsan_mutex_post_unlock(const_cast<void*>(mutex_address), __tsan_mutex_read_lock);
}

/** One half of a raw happens-before edge on @p address - what keeps a wait-free
 *  probe's pairing with the unlock releases visible once brackets hide the atomics.
 *  Call @c annotate_release before an unlock bracket opens, never inside one. */
inline void annotate_acquire(void const volatile* address) noexcept { __tsan_acquire(const_cast<void*>(address)); }

/** The releasing half of @ref annotate_acquire. */
inline void annotate_release(void const volatile* address) noexcept { __tsan_release(const_cast<void*>(address)); }

/** Reports a bug if the calling thread still holds any annotated mutex - the
 *  scenario-tail check of the concurrent suites. */
inline void assert_no_locks_held() noexcept { __tsan_check_no_mutexes_held(); }

/** No container poisoning under TSan: @c libtsan does not export the verb. */
inline void annotate_container(void const*, void const*, void const*, void const*) noexcept {}

#elif USEARCH_SANITIZE_ADDRESS

inline constexpr bool thread_sanitizer_k = false;
inline constexpr bool address_sanitizer_k = true;

inline void annotate_lock_started(void const volatile*) noexcept {}
inline void annotate_lock_acquired(void const volatile*) noexcept {}
inline void annotate_try_lock_started(void const volatile*) noexcept {}
inline void annotate_try_lock_acquired(void const volatile*) noexcept {}
inline void annotate_try_lock_refused(void const volatile*) noexcept {}
inline void annotate_unlock_started(void const volatile*) noexcept {}
inline void annotate_unlock_finished(void const volatile*) noexcept {}
inline void annotate_lock_shared_started(void const volatile*) noexcept {}
inline void annotate_lock_shared_acquired(void const volatile*) noexcept {}
inline void annotate_unlock_shared_started(void const volatile*) noexcept {}
inline void annotate_unlock_shared_finished(void const volatile*) noexcept {}
inline void annotate_acquire(void const volatile*) noexcept {}
inline void annotate_release(void const volatile*) noexcept {}
inline void assert_no_locks_held() noexcept {}

/** Re-poisons a flat container's capacity slack: @p storage_begin & @p storage_end
 *  span one owned allocation, the used prefix ends at @p old_used_end and moves to
 *  @p new_used_end. Unpoison before constructing into slack, destroy before poisoning
 *  it, and restore full addressability before the storage returns to the allocator. */
inline void annotate_container(void const* storage_begin, void const* storage_end, //
                               void const* old_used_end, void const* new_used_end) noexcept {
    if (storage_begin == storage_end) return;
    __sanitizer_annotate_contiguous_container(storage_begin, storage_end, old_used_end, new_used_end);
}

#else

inline constexpr bool thread_sanitizer_k = false;
inline constexpr bool address_sanitizer_k = false;

inline void annotate_lock_started(void const volatile*) noexcept {}
inline void annotate_lock_acquired(void const volatile*) noexcept {}
inline void annotate_try_lock_started(void const volatile*) noexcept {}
inline void annotate_try_lock_acquired(void const volatile*) noexcept {}
inline void annotate_try_lock_refused(void const volatile*) noexcept {}
inline void annotate_unlock_started(void const volatile*) noexcept {}
inline void annotate_unlock_finished(void const volatile*) noexcept {}
inline void annotate_lock_shared_started(void const volatile*) noexcept {}
inline void annotate_lock_shared_acquired(void const volatile*) noexcept {}
inline void annotate_unlock_shared_started(void const volatile*) noexcept {}
inline void annotate_unlock_shared_finished(void const volatile*) noexcept {}
inline void annotate_acquire(void const volatile*) noexcept {}
inline void annotate_release(void const volatile*) noexcept {}
inline void assert_no_locks_held() noexcept {}
inline void annotate_container(void const*, void const*, void const*, void const*) noexcept {}

#endif

} // namespace usearch
} // namespace unum
