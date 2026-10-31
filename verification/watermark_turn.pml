/**
 *  @file verification/watermark_turn.pml
 *  @author Ash Vardanian
 *  @date October 10, 2026
 *  @brief The watermark turn every dense index takes once its rows are placed.
 *
 *  A wait until the watermark holds the turn's first ID, acquired, then one store that moves it past
 *  the turn's last row. Turns go strictly in dispensed order, so the prefix below the watermark
 *  stays hole-free. The wait is the uncapped `wait_on` of `monitor_wait.pml`. A component includes
 *  this file after the memory model and passes its watermark word, its store order, and a scratch
 *  for the loads.
 */
#include <monitor_wait.pml>

// The knob: the wake-up every store to a monitored watermark gives its armed waiters.
#ifndef watermark_wakes
#define watermark_wakes true
#endif

// the wait: @p word acquired until it holds @p from, armed on its monitor between loads
inline await_turn(t, word, from, watermark) {
    do
    :: load(t, word, order_acquire, watermark);
        if
        :: watermark == from -> break
        :: else -> wait_on(t, word, watermark, false)
        fi
    od
}

// the store that ends a turn: @p to with @p order, waking the turns armed on @p word
inline end_turn(t, word, order, to) {
    atomic {
        store(t, word, order, to);
        if
        :: watermark_wakes -> wake(word)
        :: else
        fi
    }
}

// the turn from @p from to @p to: the wait, then the store that ends it
inline take_turn(t, word, from, to, order, watermark) {
    await_turn(t, word, from, watermark);
    end_turn(t, word, order, to)
}
