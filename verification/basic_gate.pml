/**
 *  The unique side of `basic::shared_mutex` from `include/usearch/basic/ram_index.hpp`, as the
 *  basic index's stop-the-world calls take it: `try_lock`'s acquire compare-exchange, only from a
 *  free word, and `unlock`'s release clear. A component includes this file after the memory model
 *  and passes its gate word.
 */

// The gate's fields: two holder bits, the unique bit, the generation above.
#define holders_of(word) ((word) & 3)
#define unique_of(word) (((word) >> 2) & 1)
#define unique_bit 4
#define generation_unit 8
#define stepped(word) ((word) + generation_unit)

// try_lock: only from a genuinely free word, a refusal is the answer
inline try_lock(t, word, observed, succeeded) {
    load(t, word, order_relaxed, observed);
    succeeded = false;
    do
    :: holders_of(observed) == 0 && !unique_of(observed) ->
        compare_exchange(t, word, order_acquire, observed, stepped(observed) | unique_bit,
                         succeeded);
        if :: succeeded -> break :: else fi
    :: else -> break
    od
}

// unlock: the unique bit cleared with a release
inline unlock(t, word, observed) {
    read_modify_write(t, word, order_release, observed, observed - unique_bit)
}
