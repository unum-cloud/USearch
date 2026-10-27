/**
 *  @file verification/dense_vram_indexes/batches.pml
 *  @author Ash Vardanian
 *  @date October 10, 2026
 *  @brief Two batches and a searcher over the shards: claims in shard order, keys before the turn,
 *      and a commit across the shards inside one sequence word.
 *
 *  `plan_insert` claims every shard's share in shard order, each claim refused while another batch
 *  is in flight on the shard, so a shard holds one batch from its claim to its turn, and so does its
 *  key column's cover; a refused claim gives the earlier shares back. Searches bound on each
 *  shard's watermark, which a batch turns on every shard inside one commit of a counted sequence
 *  word, as on the RAM shards.
 *
 *  @verify pass sc,rc11
 *  @verify pass sc quiescent_claim=false: the sharded layer holds even with claims side by side,
 *      exact as compare-exchanges make them
 *  @verify fail sc,rc11 key_before_turn=false: after the commit, the searcher reads an unset key
 *      under every memory model
 *  @verify fail rc11 watermark_order=order_relaxed: relaxed, a snapshot taken between the commit's
 *      two turns passes the re-read torn
 *  @verify fail sc commit_sequence=false: unbracketed, a snapshot taken between a commit's two turns
 *      holds one shard's part
 */
#define thread_count 3
#define location_count 7
#define history_depth 5
#define mutators 2
#include <dense_vram_indexes/protocol.pml>

proctype mutator(byte t) { insert_batch(t) }

proctype searcher(byte t) { search(t) }

init { atomic { run mutator(0); run mutator(1); run searcher(2) } }
