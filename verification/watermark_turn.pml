/**
 *  The watermark turn every dense index takes once its rows land: a wait until the watermark holds
 *  the turn's first ID, acquired, then one store that moves it past the turn's last row. Turns go
 *  strictly in dispensed order, so the prefix below the watermark stays hole-free. A component
 *  includes this file after the memory model and passes its watermark word, its store order, and
 *  a scratch for the loads.
 */

// the wait: @p word acquired until it holds @p from, blocking on its newest value between loads
inline await_turn(t, word, from, watermark) {
    (newest_value(word) == from);
    do
    :: load(t, word, order_acquire, watermark);
        if
        :: watermark == from -> break
        :: else
        fi
    od
}

// the turn from @p from to @p to: the wait, then the store with @p order
inline take_turn(t, word, from, to, order, watermark) {
    await_turn(t, word, from, watermark);
    store(t, word, order, to)
}
