/**
 *  @brief An absorb planning against a full donor beside an insert into it that crosses the shard
 *      bound: the seal `close` sets keeps a generation from opening after the packing.
 *
 *  The donor's one shard holds its one row. The absorber closes the donor at plan time under its
 *  unique side, through the claim-aware `close` over every shard the count publishes and then the
 *  seal on the count, packs its landings against the count and the watermarks, and at `run` finds
 *  every row the donor holds among them. The inserter, holding the donor's gate shared, is refused
 *  for room, opens the next generation unless the count is sealed, and inserts there. The target and
 *  the donor's emptying are left out: a packer opens target generations past the seal through its
 *  own call, and the emptying is `clearing_claims.pml`'s window.
 *
 *  @verify pass sc
 *  @verify fail sc seal=false: `close` seals the count, and `open_generation_` refuses a sealed one;
 *      unsealed, an insert arriving after the packing finds the closed shard full, opens a fresh
 *      generation whose shard is open, and lands a row on a donor shard the landings never packed
 */
#define thread_count 2
#define location_count 5
#define history_depth 1
#include "protocol.pml"

proctype inserter(byte t) {
    int observed, count, seen, watermark;
    byte shard, id;
    bool granted, won, refused;
    insert_row(t)
}

proctype absorber(byte t) {
    int observed, count, seen, packed_count, packed_rows[shards];
    byte shard;
    bool succeeded, refused;
    absorb(t)
}

init {
    atomic {
        seed(shards_count, 1);
        seed(watermark(0), row_limit);
        claimed[0] = row_limit;
        dispensed[0] = row_limit;
        run inserter(0);
        run absorber(1)
    }
}
