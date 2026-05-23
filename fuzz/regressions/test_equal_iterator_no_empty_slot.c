// Regression for fix/fuzz-equal-iterator-no-empty-slot.
//
// `equal_iterator_gt::operator++` walked the linear-probe chain until it
// hit an empty slot or another match. After enough adds with the same
// key in a multi-vector index, the slot lookup ended up with every slot
// populated (some matching, some not), so the iterator cycled through
// the matches forever and `std::distance(begin, end)` inside
// `usearch_remove` hung. libFuzzer reported a timeout.
//
// The fix tracks both the iteration start and a cumulative-step counter
// across `++` calls and pins the iterator to a `capacity_slots_`
// sentinel after one full pass.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "usearch.h"

int main(void) {
    usearch_init_options_t opts;
    memset(&opts, 0, sizeof(opts));
    opts.metric_kind = usearch_metric_cos_k;
    opts.quantization = usearch_scalar_bf16_k;
    opts.dimensions = 1;
    opts.connectivity = 17;
    opts.expansion_add = 16;
    opts.expansion_search = 16;
    opts.multi = true;

    usearch_error_t error = NULL;
    usearch_index_t index = usearch_init(&opts, &error);
    if (!index || error) {
        fprintf(stderr, "FAIL: init: %s\n", error ? error : "(null)");
        return 1;
    }

    usearch_reserve(index, 5, &error);
    error = NULL;

    // Add 150 vectors all with the same key — saturates the slot lookup
    // with identical-key entries.
    float v[1] = {-1.0f};
    for (int i = 0; i < 150; ++i) {
        usearch_add(index, /*key=*/0, v, usearch_scalar_f32_k, &error);
        error = NULL;
    }

    // Now attempt to remove a different key. Pre-fix the iterator inside
    // `equal_range` for the no-empty-slot table cycled forever.
    clock_t start = clock();
    usearch_remove(index, /*key=*/55, &error);
    clock_t elapsed = clock() - start;
    double seconds = (double)elapsed / (double)CLOCKS_PER_SEC;
    if (seconds > 2.0) {
        fprintf(stderr, "FAIL: remove took %.2fs — likely the iterator infinite loop\n", seconds);
        return 1;
    }

    // And remove key=0 itself — should also terminate quickly.
    start = clock();
    usearch_remove(index, /*key=*/0, &error);
    elapsed = clock() - start;
    seconds = (double)elapsed / (double)CLOCKS_PER_SEC;
    if (seconds > 2.0) {
        fprintf(stderr, "FAIL: remove(0) took %.2fs — likely the iterator infinite loop\n", seconds);
        return 1;
    }

    usearch_free(index, &error);
    printf("PASS: equal_iterator_no_empty_slot regression\n");
    return 0;
}
