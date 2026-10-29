/**
 *  @file verification/dense_vram_indexes/batches.pml
 *  @author Ash Vardanian
 *  @date October 10, 2026
 *  @brief Two batches and a searcher over the shards: claims in shard order, keys uploaded ahead of
 *      the launch and filed after the drain, and a commit across the shards inside one sequence word.
 *
 *  `plan_insert` claims every shard's share in shard order, each claim refused while another batch
 *  is in flight on the shard, so a shard holds one batch from its claim to its turn, and its map one
 *  writer; a refused claim gives the earlier shares back. Each launch may fail its drain, keeping its
 *  failure and its claim. Searches bound on each shard's watermark, which a batch turns on every
 *  shard inside one commit of a counted sequence word, as on the RAM shards.
 *
 *  - Once all are done, no row of a failed batch is filed.
 *
 *  @verify pass sc,rc11
 *  @verify fail sc quiescent_claim=false: two batches in flight on one shard both file into its map
 *      before their turns, and the second takes the slot the first is filing
 *  @verify fail rc11 claim_order=order_relaxed: a claim that reads the watermark relaxed carries no
 *      view of the previous holder's filings, and its relaxed peek files over one of them
 *  @verify fail sc keys_after_drain=false: filed at the launch, a failed drain leaves its rows in the
 *      map, under a watermark that never reaches them
 *  @verify fail sc,rc11 key_before_turn=false: after the commit, the searcher reads an unset key
 *      under every memory model
 *  @verify fail rc11 watermark_order=order_relaxed: relaxed, a snapshot taken between the commit's
 *      two turns passes the re-read torn
 *  @verify fail sc commit_sequence=false: unbracketed, a snapshot taken between a commit's two turns
 *      holds one shard's part
 */
#define thread_count 3
#define location_count 11
#define history_depth 5
#define mutators 2
#include <dense_vram_indexes/protocol.pml>

proctype mutator(byte t) { insert_batch(t) }

proctype searcher(byte t) { search(t) }

init {
    atomic { run mutator(0); run mutator(1); run searcher(2) };
    (done == 3);
    audit_keys()
}
