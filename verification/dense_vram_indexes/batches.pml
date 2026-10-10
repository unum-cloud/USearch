/**
 *  @brief Two batches and a searcher over the shards: the mutation gate, keys before the turn, and
 *      a commit across the shards inside one sequence word.
 *
 *  `insert` takes the executor's hold and then `mutating_`, so two batches never interleave their
 *  claims, and the gate serializes the key columns' covers along with them, so the batches admit
 *  no rows against the key planes. Searches bound on each shard's watermark, which a batch turns
 *  on every shard inside one commit of a counted sequence word, as on the RAM shards.
 *
 *  @verify pass sc,rc11
 *  @verify fail sc gate=false: `mutating_` serializes the batches' claims; without it, the
 *      mutators interleave their claims, and one writes its key over a row the other's claim
 *      confirmed
 *  @verify fail sc,rc11 key_before_turn=false: each key lands before the shard's claim and the
 *      commit; after the commit, the searcher reads an unset key under every memory model
 *  @verify fail rc11 watermark_order=order_relaxed: the turn's release carries more than the keys,
 *      it brings the commit's opening add on the sequence word to the snapshot's re-read; relaxed,
 *      a snapshot taken between the commit's two turns passes the re-read torn
 *  @verify fail sc commit_sequence=false: the searcher brackets its snapshot by two equal reads of
 *      the sequence word with no commit in flight; unbracketed, a snapshot taken between a
 *      commit's two turns holds one shard's part
 */
#define thread_count 3
#define location_count 8
#define history_depth 5
#define mutators 2
#include "protocol.pml"

proctype mutator(byte t) { insert_batch(t) }

proctype searcher(byte t) { search(t) }

init { atomic { run mutator(0); run mutator(1); run searcher(2) } }
