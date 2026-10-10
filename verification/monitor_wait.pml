/**
 *  @file verification/monitor_wait.pml
 *  @author Ash Vardanian
 *  @date October 10, 2026
 *  @brief A wait that blocks until a write wakes it, so a lost wake-up leaves its waiter blocked
 *      forever and the search reports an invalid end state.
 *
 *  A waiting loop loads its word and leaves once the word moved; otherwise it calls @c wait_on with
 *  the value it read. The wait arms the thread on the word, re-checks the newest write when
 *  @c recheck holds, as a monitored waiter's load after arming does and as a futex re-reads the
 *  word under its lock, and then blocks until a write wakes it or its cap lets it go. A write
 *  wakes every thread armed on its word only where the model calls @c wake after it: after every
 *  write to the word for a monitor-style wait, since any store to a monitored line clears the
 *  monitor, and only at the notifying sites for a futex-style one, whose plain stores wake no one.
 *
 *  The re-check reads the newest write rather than any write in the thread's view: arming takes
 *  the line, so the load after it is as new as the arming, and the caller's own load on its next
 *  turn decides what the word says and what it synchronizes with. Without the re-check, a write
 *  that lands between the caller's load and the arming wakes nobody, and an uncapped waiter waits
 *  for a write that already happened.
 *
 *  A capped wait also returns once its cap condition holds: whatever else its loop guards, read as
 *  newest values, since a timeout bounds how late the loop notices it. An uncapped wait passes
 *  @c false. No wait ever returns spuriously, so a protocol that relies on a spurious wake-up
 *  shows as blocked.
 *
 *  Include after `weak_memory.pml`, whose @c thread_count and @c newest_value this file uses, and
 *  before the protocol that waits.
 */

/** The knob: the waiter's re-check after arming, which every monitored waiter and every futex
 *  makes; overridden by a `@verify` line. */
#ifndef recheck
#define recheck true
#endif

byte armed[thread_count]; // one past the location the thread's wait is armed on, or zero
bool event[thread_count]; // a write to the armed location woke the thread since it armed
hidden byte woken;        // scratch for the wake loop

/** Wakes every thread whose wait is armed on location @p l: a store to a monitored line, or the
 *  notify after a futex-style write. */
inline wake(l) {
    atomic {
        for (woken : 0 .. thread_count - 1) {
            if
            :: armed[woken] == (l) + 1 -> event[woken] = true
            :: else
            fi
        }
    }
}

/** Thread @p t waits on location @p l, which its loop read as @p observed: arms, re-checks the
 *  newest write if @c recheck, and blocks until a @c wake or until @p capped holds. One step when
 *  it does not block; when it does, others run while it waits, armed. */
inline wait_on(t, l, observed, capped) {
    atomic {
        armed[t] = (l) + 1;
        event[t] = recheck && newest_value(l) != (observed);
        (event[t] || (capped));
        armed[t] = 0;
        event[t] = false
    }
}
