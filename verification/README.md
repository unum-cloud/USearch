# Verification

Model checking for the index protocols: the gates, the watermarks, the capacity counters, the public calls interleaved, the CUDA index's rows and the NVMe index's rings.
[Spin](https://spinroot.com) checks each protocol as a Promela model, and [GenMC](https://github.com/MPI-SWS/genmc) checks the C++ against the real header where it fits.

- `basic_gate.pml`: the unique side of `basic::shared_mutex`, as the stop-the-world calls take it.
- `basic_shared_mutex.pml`, `basic_shared_mutex.cpp`: the one-word gate every mutator holds shared and the certified searches bracket with two probes.
- `basic_ram_index_rows.pml`: the basic index's rows under their lock word, behind the gate's shared and unique sides.
- `basic_ram_index_back_link.pml`: the back-link re-selection on one full row, against `link` and `unlink`.
- `basic_ram_index_steps.pml`: the basic load path, two steps' tally, then `finish` under the unique side.
- `basic_ram_index_batches/`: a basic batch, landed serially or planned over steppers, publishes whole.
- `watermark_turn.pml`: the watermark turn every dense index takes once its rows land.
- `dense_ram_index_watermark/`: the dense index's publication protocol, with its cursor, save, load and refinement.
- `dense_ram_index_batches.pml`, `dense_ram_index_batches.cpp`: a batch's commit, one add for its range and one watermark turn.
- `dense_ram_index_capacity/`: storage capacity grown on demand behind one elected grower.
- `dense_ram_indexes_claims.pml`: the sharded index's claims, beside a snapshot.
- `dense_ram_indexes_commits.pml`: a batch's commit across the shards, behind one counted sequence word.
- `dense_ram_indexes_cover.pml`: adders covering their key column before they dispense.
- `dense_ram_indexes_plans/`: the sharded save and load plans.
- `dense_ram_indexes_generations/`: the shard set's generations, each opened by one caller.
- `dense_ram_indexes_gate/`: the sharded index's gate and epoch, against clears, absorbs, cursors and compaction.
- `dense_vram_index_claims.pml`: the CUDA index's host side, the claim, the growth, the turn and the admission ramp.
- `dense_vram_index_locks.pml`: the CUDA index's rows, published by one release under the bounded slot-zero lock.
- `dense_vram_index_merge.pml`: the CUDA refine merging a re-selected row into the live one.
- `dense_vram_indexes/`: the sharded CUDA index's batches, load and save.
- `nvme_ring.pml`: the io_uring words of `linux_uring_ring_t` behind the ring engines.
- `ram_tables.pml`, `ram_tables_growth.pml`: `chunked_multimap`'s probes and its chunk growth.

## Running

`./check.sh` runs every model, and `./check.sh <file>...` only those; the GenMC clients run when `GENMC` names its executable.
The runner comes from the `forkunion` submodule, `git submodule update --init forkunion`, or from the checkout `FORKUNION_SOURCE` names.
ForkUnion's [verification README](https://github.com/ashvardanian/ForkUnion/blob/main-v4/verification/README.md) documents the runner, the `@verify` grammar, the include roots and the memory budgets.

## What the Models Found

- `basic::shared_mutex::lock_shared` ends in a release fence, or a certified search certifies a torn pair of rows: [`basic_shared_mutex.pml`](basic_shared_mutex.pml), [`basic_shared_mutex.cpp`](basic_shared_mutex.cpp).
- The gate's back-off wait and the deficit wait in `claim_capacity` are capped: [`basic_shared_mutex.pml`](basic_shared_mutex.pml).
- The gate's generation wrap is priced at days of churn: [`basic_shared_mutex.pml`](basic_shared_mutex.pml).
- `unlock_shared`'s release and `try_lock`'s acquire carry the shared side's edits to the unique side: [`basic_shared_mutex.pml`](basic_shared_mutex.pml).
- The basic index's rows carry their edits on the row lock's acquire and the unlock's release: [`basic_ram_index_rows.pml`](basic_ram_index_rows.pml).
- A walk admits another batch's row only once it acquires that row's `wired_k` bit: [`basic_ram_index_batches/serial.pml`](basic_ram_index_batches/serial.pml).
- `finish` refuses until every position landed, and a refused fan-out lands on the caller: [`basic_ram_index_batches/planned.pml`](basic_ram_index_batches/planned.pml).
- `back_link_` revalidates the relocked row slot by slot: [`basic_ram_index_back_link.pml`](basic_ram_index_back_link.pml).
- The load path's `finish` stores the counters before the watermark's release: [`basic_ram_index_steps.pml`](basic_ram_index_steps.pml).
- A failed load wipes the keys it registered, and `insert` clears its row before wiring: [`test/basic/sentences.cpp`](../test/basic/sentences.cpp).
- `chunked_multimap` publishes a slot with a release and probes it with an acquire: [`ram_tables.pml`](ram_tables.pml), [`ram_tables_growth.pml`](ram_tables_growth.pml).
- A `concurrent_k` save copies each row under its slot-zero lock: [`dense_ram_index_watermark/refining.pml`](dense_ram_index_watermark/refining.pml).
- A load or a refinement closes `next_dispensed_id_`, and every dispense refuses while it is closed: [`dense_ram_index_watermark/closing.pml`](dense_ram_index_watermark/closing.pml).
- A cursor's `seek` refuses an ID at or past its snapshot, and `refresh` acquires the watermark: [`dense_ram_index_watermark/cursor.pml`](dense_ram_index_watermark/cursor.pml).
- The load path's `finish` moves an open dispenser by one compare-exchange from its base: [`dense_ram_index_watermark/appender.pml`](dense_ram_index_watermark/appender.pml).
- A walk admits a row past its snapshot only by an acquire of its slot zero: [`dense_ram_index_batches.pml`](dense_ram_index_batches.pml).
- A turn's waiters re-read the watermark after arming, and the store that ends a turn wakes them: [`dense_ram_index_batches.pml`](dense_ram_index_batches.pml).
- `release_capacity`'s subtract releases: [`dense_ram_index_capacity/`](dense_ram_index_capacity/).
- A release lands only on a whole counter and the grower's take releases: [`dense_ram_index_capacity/reservers.pml`](dense_ram_index_capacity/reservers.pml).
- A second `reserve` above the physical rows claims up to its request: [`test/dense.cuh`](../test/dense.cuh).
- A planned batch claims every shard before it dispenses: [`dense_ram_indexes_claims.pml`](dense_ram_indexes_claims.pml).
- Every claimer covers the shard's key column before dispensing: [`dense_ram_indexes_cover.pml`](dense_ram_indexes_cover.pml).
- The monolith save holds every batch on all of its shards or on none: [`dense_ram_indexes_plans/`](dense_ram_indexes_plans/).
- A batch across shards stores its watermarks behind a counted sequence word: [`dense_ram_indexes_commits.pml`](dense_ram_indexes_commits.pml).
- The sharded index closes its shards' dispensers in placement order and reopens them the last first: [`dense_ram_indexes_commits.pml`](dense_ram_indexes_commits.pml).
- A shard generation opens behind one compare-exchange of the shard count to its opening bit: [`dense_ram_indexes_generations/`](dense_ram_indexes_generations/).
- `close` refuses a shard holding claims it has not dispensed: [`dense_ram_indexes_gate/clearing_claims.pml`](dense_ram_indexes_gate/clearing_claims.pml).
- Every mutator holds the gate's shared side for its call: [`dense_ram_indexes_gate/clearing_locators.pml`](dense_ram_indexes_gate/clearing_locators.pml).
- An absorb closes and seals its donors at plan time: [`dense_ram_indexes_gate/absorbing_seal.pml`](dense_ram_indexes_gate/absorbing_seal.pml), [`dense_ram_indexes_gate/absorbing_donors.pml`](dense_ram_indexes_gate/absorbing_donors.pml).
- A destructive window makes the epoch odd with a relaxed add and a release fence: [`dense_ram_indexes_gate/epoch.pml`](dense_ram_indexes_gate/epoch.pml).
- A cursor's read re-checks the epoch it recorded at construction: [`dense_ram_indexes_gate/cursor.pml`](dense_ram_indexes_gate/cursor.pml).
- In-place compaction's `replace` keeps every key in its slot: [`dense_ram_indexes_gate/compacting.pml`](dense_ram_indexes_gate/compacting.pml).
- The CUDA index's host side claims by one compare-exchange from the watermark: [`dense_vram_index_claims.pml`](dense_vram_index_claims.pml).
- The CUDA rows need a `__syncwarp()` after each shuffle and one release fence per lock hold: [`dense_vram_index_locks.pml`](dense_vram_index_locks.pml).
- The sharded CUDA index closes every shard around its load and its save: [`dense_vram_indexes/`](dense_vram_indexes/).
- A refine's merge never stores the sentinel into slot zero: [`dense_vram_index_merge.pml`](dense_vram_index_merge.pml).
- `abandon_` waits out every read the kernel consumed before it drops their completions: [`nvme_ring.pml`](nvme_ring.pml).

## What Is Not Covered

- The dense header's first chunk is more events than GenMC's execution graph holds, so the dense clients spell the counters' lines.
- The far-memory results are Spin's alone: GenMC has no posted add.
- `chunked_multimap` crashed GenMC 0.18's interpreter before the first insert, so the multimap has no client, and 0.19 is untried.
- The rows model has no client either: the row lock sits inside `basic::ram_index`.
- The executors' holds are plain try-locks around the per-thread arenas and are not modelled.
- IMM may admit more than real Arm and POWER chips do, and nothing here runs on silicon.
- The CUDA rows model folds the prior lock holder into one lane and takes `bar.warp.sync`'s ordering among its participants from the ISA text.
- Litmus tests for the instruction-set references belong beside ForkUnion's probes, and are the next layer down.
