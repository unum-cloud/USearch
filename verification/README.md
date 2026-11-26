# Verification

Model checking for the index protocols: the gates, the watermarks, the capacity counters, the public calls interleaved, the CUDA index's rows and the NVMe index's rings.
[Spin](https://spinroot.com) checks each protocol as a Promela model, and [GenMC](https://github.com/MPI-SWS/genmc) checks the C++ against the real header where it fits.

- `basic_gate.pml`: the unique side of `basic::shared_mutex`, as the stop-the-world calls take it.
- `basic_shared_mutex.pml`, `basic_shared_mutex.cpp`: the one-word gate every mutator holds shared and the certified searches bracket with two probes.
- `basic_ram_index_rows.pml`: the basic index's rows under their lock word, behind the gate's shared and unique sides.
- `basic_ram_index_back_link.pml`: the back-link re-selection on one full row, against `link` and `unlink`.
- `basic_ram_index_steps.pml`: the basic load path, two steps' tally, then `finish` under the unique side.
- `basic_ram_index_batches/`: a basic batch, placed serially or planned over steppers, on one index or across two shards, publishes whole.
- `watermark_turn.pml`: the watermark turn every dense index takes once its rows are placed.
- `dense_ram_index_watermark/`: the dense mono-index's publication protocol, with its keys, cursor, save, load and refinement.
- `dense_ram_index_batches.pml`, `dense_ram_index_batches.cpp`: a batch's commit, one add for its range and one watermark turn.
- `dense_ram_index_capacity/`: storage capacity grown on demand behind one elected grower.
- `dense_ram_index_refine.pml`: a refinement round's `rewrite_row_` relocking a row while a search walks it.
- `dense_ram_indexes_claims.pml`: the sharded index's claims, beside a snapshot.
- `dense_ram_indexes_commits.pml`: a batch's commit across the shards, behind one counted sequence word.
- `dense_ram_indexes_plans/`: the sharded save and load plans.
- `dense_ram_indexes_generations/`: the shard set's generations, each opened by one caller, with batches crossing and spanning the bound.
- `dense_ram_indexes_gate/`: the sharded index's gate and epoch, against clears, absorbs, cursors and compaction, the MaxSim scan's leads over a snapshot, and a walk beside a clear.
- `dense_vram_index_claims.pml`: the CUDA index's host side, the claim, the growth, the turn and the admission ramp.
- `dense_vram_index_locks.pml`: the CUDA index's rows, published by one release under the bounded slot-zero lock.
- `dense_vram_index_merge.pml`: the Apple refine merging a re-selected row into the live one.
- `dense_vram_indexes/`: the sharded CUDA index's batches, load and save, and the await of a search across its shards' streams.
- `arena_lifetime.pml`: an arena's calls, generation and exports, its free and its executor's free, the same file in every engine.
- `nvme_ring.pml`: the io_uring words of `linux_uring_ring_t` behind the ring engines.
- `ram_tables.pml`, `ram_tables_growth.pml`: `chunked_multimap`'s probes and its chunk growth.

## Running

`./check.sh` runs every model, and `./check.sh <file>...` only those; the GenMC clients run when `GENMC` names its executable.
The runner comes from the `forkunion` submodule, `git submodule update --init forkunion`, or from the checkout `FORKUNION_SOURCE` names.
ForkUnion's [verification README](https://github.com/ashvardanian/ForkUnion/blob/main/verification/README.md) documents the runner, the `@verify` grammar, the include roots and the memory budgets.

## What the Models Found

- `basic::shared_mutex::lock_shared` ends in a release fence, or a certified search certifies a torn pair of rows: [`basic_shared_mutex.pml`](basic_shared_mutex.pml), [`basic_shared_mutex.cpp`](basic_shared_mutex.cpp).
- The gate's back-off wait and the deficit wait in `claim_capacity` are capped: [`basic_shared_mutex.pml`](basic_shared_mutex.pml).
- The gate's generation wrap is priced at days of churn: [`basic_shared_mutex.pml`](basic_shared_mutex.pml).
- `unlock_shared`'s release and `try_lock`'s acquire publish the shared side's edits to the unique side: [`basic_shared_mutex.pml`](basic_shared_mutex.pml).
- The basic index's rows publish their edits through the row lock's acquire and the unlock's release: [`basic_ram_index_rows.pml`](basic_ram_index_rows.pml).
- A walk admits another batch's row only once it acquires that row's `wired_k` bit: [`basic_ram_index_batches/serial.pml`](basic_ram_index_batches/serial.pml).
- `finish` refuses until every position is placed, and a refused fan-out places on the caller: [`basic_ram_index_batches/planned.pml`](basic_ram_index_batches/planned.pml).
- A basic batch across two shards publishes whole, even when two batches cross between them: [`basic_ram_index_batches/sharded.pml`](basic_ram_index_batches/sharded.pml).
- `back_link_` revalidates the relocked row slot by slot: [`basic_ram_index_back_link.pml`](basic_ram_index_back_link.pml).
- The load path's `finish` stores the counters before the watermark's release: [`basic_ram_index_steps.pml`](basic_ram_index_steps.pml).
- A failed load wipes the keys it registered, and `insert` clears its row before wiring: [`test/basic/sentences.cpp`](../test/basic/sentences.cpp).
- `chunked_multimap` publishes a slot with a release and probes it with an acquire: [`ram_tables.pml`](ram_tables.pml), [`ram_tables_growth.pml`](ram_tables_growth.pml).
- A `concurrent_k` save copies each row under its slot-zero lock: [`dense_ram_index_watermark/refining.pml`](dense_ram_index_watermark/refining.pml).
- A load or a refinement closes `next_dispensed_id_`, and every dispense refuses while it is closed: [`dense_ram_index_watermark/closing.pml`](dense_ram_index_watermark/closing.pml).
- A cursor's `seek` refuses an ID at or past its snapshot, and `refresh` acquires the watermark: [`dense_ram_index_watermark/cursor.pml`](dense_ram_index_watermark/cursor.pml).
- The load path's `finish` moves an open dispenser by one compare-exchange from its base: [`dense_ram_index_watermark/appender.pml`](dense_ram_index_watermark/appender.pml).
- `keyed_nodes::assign` writes a row's key right after the dispense, and the turn holder publishes it into the map before the watermark: [`dense_ram_index_watermark/keyed.pml`](dense_ram_index_watermark/keyed.pml).
- A walk reads a row a round rewrites slot by slot old or merged, mixed when it peeked the old slot zero, and whole once it acquires the merged one: [`dense_ram_index_refine.pml`](dense_ram_index_refine.pml).
- A walk admits a row past its snapshot only by an acquire of its slot zero: [`dense_ram_index_batches.pml`](dense_ram_index_batches.pml).
- A turn's waiters re-read the watermark after arming, and the store that ends a turn wakes them: [`dense_ram_index_batches.pml`](dense_ram_index_batches.pml).
- `release_capacity`'s subtract releases: [`dense_ram_index_capacity/`](dense_ram_index_capacity/).
- A release applies only on a whole counter and the grower's take releases: [`dense_ram_index_capacity/reservers.pml`](dense_ram_index_capacity/reservers.pml).
- A second `reserve` above the physical rows claims up to its request: [`test/dense.cuh`](../test/dense.cuh).
- A planned batch claims every shard before it dispenses: [`dense_ram_indexes_claims.pml`](dense_ram_indexes_claims.pml).
- The monolith save holds every batch on all of its shards or on none: [`dense_ram_indexes_plans/`](dense_ram_indexes_plans/).
- `publish_keys` files a shard's keys only once every row arrived, so a refused `finish` leaves no key above the watermark: [`dense_ram_indexes_plans/loading.pml`](dense_ram_indexes_plans/loading.pml), [`dense_vram_indexes/loading.pml`](dense_vram_indexes/loading.pml).
- A batch across shards stores its watermarks behind a counted sequence word: [`dense_ram_indexes_commits.pml`](dense_ram_indexes_commits.pml).
- The sharded index closes its shards' dispensers in placement order and reopens them the last first: [`dense_ram_indexes_commits.pml`](dense_ram_indexes_commits.pml).
- A shard generation opens behind one compare-exchange of the shard count to its opening bit: [`dense_ram_indexes_generations/`](dense_ram_indexes_generations/).
- A batch crossing or spanning the generation bound publishes on every shard the count publishes: [`dense_ram_indexes_generations/crossing.pml`](dense_ram_indexes_generations/crossing.pml), [`dense_ram_indexes_generations/spanning.pml`](dense_ram_indexes_generations/spanning.pml).
- `close` refuses a shard holding claims it has not dispensed: [`dense_ram_indexes_gate/clearing_claims.pml`](dense_ram_indexes_gate/clearing_claims.pml).
- An insert plan's `finish` files its keys into the shard's `keyed_nodes` inside the watermark turn, so no key names a row a `clear` forgot: [`dense_ram_indexes_gate/clearing_locators.pml`](dense_ram_indexes_gate/clearing_locators.pml).
- A walk reaching a row a `clear` forgot notices the watermark dropped beneath it rather than wait on its lock: [`dense_ram_indexes_gate/forgetting_walks.pml`](dense_ram_indexes_gate/forgetting_walks.pml).
- The MaxSim scan's `leads_` nominate each key at exactly one row of a snapshot: [`dense_ram_indexes_gate/leading.pml`](dense_ram_indexes_gate/leading.pml).
- An absorb closes and seals its donors at plan time: [`dense_ram_indexes_gate/absorbing_seal.pml`](dense_ram_indexes_gate/absorbing_seal.pml), [`dense_ram_indexes_gate/absorbing_donors.pml`](dense_ram_indexes_gate/absorbing_donors.pml).
- A destructive window makes the epoch odd with a relaxed add and a release fence: [`dense_ram_indexes_gate/epoch.pml`](dense_ram_indexes_gate/epoch.pml).
- A cursor's read re-checks the epoch it recorded at construction: [`dense_ram_indexes_gate/cursor.pml`](dense_ram_indexes_gate/cursor.pml).
- An in-place compaction moves an entry's key between the shards' `keyed_nodes` inside one epoch window, its `finish` and `forget_moved_` both, so a count of the key stays exact: [`dense_ram_indexes_gate/compacting.pml`](dense_ram_indexes_gate/compacting.pml).
- The CUDA index's host side claims by one compare-exchange from the watermark: [`dense_vram_index_claims.pml`](dense_vram_index_claims.pml).
- The CUDA rows need a `__syncwarp()` after each shuffle and one release fence per lock hold: [`dense_vram_index_locks.pml`](dense_vram_index_locks.pml).
- The sharded CUDA index closes every shard around its load and its save: [`dense_vram_indexes/`](dense_vram_indexes/).
- A load's column copies are awaited through the shard stream's event before its turn, after a short shard is refused: [`dense_vram_indexes/loading.pml`](dense_vram_indexes/loading.pml).
- A save reads each pinned half of its bounce only after the event its copies recorded, the pending half in `finish`: [`dense_vram_indexes/saving.pml`](dense_vram_indexes/saving.pml).
- A search's await queues every shard's event record before the first stream's wait on it, and every record releases: [`dense_vram_indexes/awaiting.pml`](dense_vram_indexes/awaiting.pml).
- A refine's merge never stores the sentinel into slot zero: [`dense_vram_index_merge.pml`](dense_vram_index_merge.pml).
- `abandon_` waits out every read the kernel consumed before it drops their completions: [`nvme_ring.pml`](nvme_ring.pml).
- An arena's free waits out the calls naming it before it reads the export count, or an export opened during the wait is open when the arena is deleted, and it counts the arena off its executor last, after the memory went back through it: [`arena_lifetime.pml`](arena_lifetime.pml).
- A call advances its arena's generation once, after the export count and the executor's claim admitted it and never on its body's re-hold, and a binding copies an output only under an export, re-reading the generation inside it, or a copy reads the next call's outputs as current: [`arena_lifetime.pml`](arena_lifetime.pml).
- A call takes the arena and an export raises the count each with a sequentially consistent read-modify-write before reading the other's word, or both read the other's from before and the call rewinds outputs the export holds: [`arena_lifetime.pml`](arena_lifetime.pml).
- A call the pool refuses after its claims has advanced the generation without rewinding, which reads earlier results as stale and never as current: [`arena_lifetime.pml`](arena_lifetime.pml).

## What Is Not Covered

- The dense header's first chunk is more events than GenMC's execution graph holds, so the dense clients spell the counters' lines.
- The far-memory results are Spin's alone: GenMC has no posted add.
- `chunked_multimap` has no client: `chunked_column` zeroes each fresh chunk with a `std::memset` of runtime length into heap memory, which GenMC 0.20 refuses to run, and its first chunk holds 2048 slots, more turns than `--unroll` gives a zeroing loop.
- The rows model has no client either: the row lock sits inside `basic::ram_index`.
- The executors' holds are plain try-locks around the per-thread arenas and are not modelled.
- IMM may admit more than real Arm and POWER chips do, and nothing here runs on silicon.
- A dying load plan's wait on its stream before the shards reopen is not modelled; `load_plan_t`'s destructor holds it.
- The CUDA rows model folds the prior lock holder into one lane and takes `bar.warp.sync`'s ordering among its participants from the ISA text.
- Litmus tests for the instruction-set references belong beside ForkUnion's probes, and are the next layer down.
