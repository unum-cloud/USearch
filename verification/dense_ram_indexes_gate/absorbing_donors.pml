/**
 *  @brief An absorb planning against a donor beside an insert into its room: the absorb closes
 *      its donors at plan time, so no row lands after the packing.
 *
 *  The donor's one shard is empty, with room for one row. The absorber closes the donor at plan
 *  time under its unique side, through the claim-aware `close` and the seal, packs its landings
 *  against the count and the watermarks, and at `run` finds every row the donor holds among them.
 *  The inserter, holding the donor's gate shared, claims the room and dispenses, or a closed
 *  dispenser refuses it and it gives the claim back. The target and the donor's emptying are left
 *  out, as in `absorbing_seal.pml`.
 *
 *  @verify pass sc
 *  @verify fail sc donor_close=at_run: the absorb closes its donors at plan time; closing them only
 *      when `run` lands them, an insert between the packing and the landing adds a row the landings
 *      miss, which `landings_of_` then reads past
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
        run inserter(0);
        run absorber(1)
    }
}
