/**
 *  @file verification/dense_vram_indexes/awaiting.pml
 *  @author Ash Vardanian
 *  @date October 30, 2026
 *  @brief Two searches over two shards, each merged on the first shard's stream and awaited by the
 *      host once: `vram_shard_fanout::await_every_shard` in `include/usearch/utilities/vram_shards.hpp`.
 *
 *  The host queues each shard's work in shard order, each context recording its event behind its
 *  submission; a later shard's stream first waits on the first shard's record, then writes its
 *  ranking block into the first shard's outputs, and records. The first stream waits on every other
 *  record, merges every block, and records the event the host sleeps on. Every event record is a
 *  release store of its generation and every wait an acquire. The second search reuses the same
 *  contexts and outputs.
 *
 *  - The merge reads every shard's block of its own search.
 *  - The host reads the rankings only after every shard's work and the merge finished.
 *  - A block is never written while the merge of an earlier search may still read the outputs.
 *
 *  @verify pass sc,rc11
 *  @verify fail sc record_before_wait=false: queued ahead of the record it names, the first stream's
 *      wait takes the event as already complete, and the merge reads a block not yet written
 *  @verify fail rc11 record_order=order_relaxed: a relaxed record carries none of its stream's
 *      writes, and the merge reads the later shard's block stale
 *  @verify fail sc host_wait_per_call=false block_waits_first_shard=false: with neither the host's
 *      wait nor the later shard's wait on the first one, the second search's block overwrites the
 *      outputs the first merge still reads
 */
#define thread_count 3
#define location_count 18
#define history_depth 5
#define mutators 1
#define first_stream 1
#define calls 2
#include <dense_vram_indexes/protocol.pml>

// the searches: each queued whole, then awaited once, or every one awaited after all are queued
proctype host(byte t) {
    byte call;
    int seen;
    for (call : 1 .. calls) {
        submit_call(call);
        if
        :: host_wait_per_call -> await_call(t, call, seen)
        :: else
        fi
    };
    if
    :: !host_wait_per_call ->
        for (call : 1 .. calls) {
            await_call(t, call, seen)
        }
    :: else
    fi;
    done++
}

// a shard's stream, running its context's work, records, waits and, on the first, the merges
proctype stream(byte t) {
    byte kind, named, value;
    int seen;
end:
    do
    :: stream_queue[t - first_stream] ? kind, named, value ->
        run_operation(t, t - first_stream, kind, named, value, seen)
    od
}

init {
    atomic { run host(0); run stream(1); run stream(2) };
    (done == 1)
}
