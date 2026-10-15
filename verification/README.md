# Verification

Model checking for the index protocols: the gates, the watermarks, the capacity counters, the public calls interleaved, the CUDA index's rows and the NVMe index's rings.
[Spin](https://spinroot.com) checks each protocol as a Promela model, and [GenMC](https://github.com/MPI-SWS/genmc) checks the C++ against the real header where it fits, under SC, RC11 and IMM, the model of compiled code on x86, Arm and POWER.

- `weak_memory.pml` is the C++ memory model as views, for Promela, under three models chosen by `-Dmemory=`: `sc`, one copy of every word; `rc11`, the default, RC11's release-acquire-relaxed fragment; and `far`, which also posts a relaxed no-return add the way RAO-INT does.
- `monitor_wait.pml` is a wait that can miss its wake-up: a waiter arms, re-checks the word, and blocks until a writer's `wake`, so a lost notify reads as `stuck`.
- `check.sh` is the runner, and `genmc.hpp` the threads and the assertion a client takes from GenMC.
- `watermark_turn.pml` and `basic_gate.pml` are protocols several components include.

## Writing a Model

A component is one mechanism of the code under test: a flat `<name>.pml` while it has one scenario, and a directory `<name>/` once it has a second, holding a `protocol.pml` with its words, values, knobs and inlines, one file per scenario, and a `client.cpp` when GenMC checks the same words.
A scenario defines its shape, `thread_count`, `location_count` and `history_depth`, includes its protocol, and starts its roles from one `init { atomic { run <role>(<thread>); … } }`; the depth is the exact minimum every `rc11` and `far` line of the scenario accepts.
There is no conditional compilation: a knob is a choice the header makes, defaulted under `#ifndef` and read as the guard of a plain `if` with an `else` branch, and memory-model differences are expressions on `memory`.
A wait on a word another thread moves goes through `monitor_wait.pml`'s `wait_on`, and the writer that ends it calls `wake`, so a dropped wake-up or a missing re-check comes out `stuck`.
A client spells the same knobs as fields of a `knobs_t`, one `constexpr knobs_t <variant>_k` per variant behind a `template <knobs_t const &knobs_>`, and one `extern "C"` entry per variant, `<scenario>` or `<scenario>_<knob>_<value>`.

Every expected verdict is a `@verify` line in the docblock of the file it checks:

```text
@verify <pass|fail|stuck> <memory>[,<memory>...] [<knob>=<value> ...][: <finding>]   in a scenario
@verify <pass|fail|stuck> <memory>[,<memory>...] <entry>[: <finding>]                in a client
```

`pass` is a search that completed with no error, `fail` a violated assertion, and `stuck` an invalid end state in Spin or a liveness violation in GenMC; a search cut short, a knob defined twice, or a write past the scenario's shape is `broken`.
Every scenario has a `pass` line, every knob appears on some line, and adding a scenario, a knob or a finding never edits `check.sh`.

## What the Models Found

Each finding is the `@verify` line that replays it, where the full counterexample is written out.

- `basic::shared_mutex::lock_shared` ends in a release fence; without it, a certified search certifies a torn pair of rows, under RC11 and under IMM on Arm and POWER, though x86's locked add orders the rows on its own: [`basic_shared_mutex.pml`](basic_shared_mutex.pml), [`basic_shared_mutex.cpp`](basic_shared_mutex.cpp).
- The gate's back-off wait and the deficit wait in `claim_capacity` are capped; uncapped, a mutator arms its monitor after the holder left and waits forever: [`basic_shared_mutex.pml`](basic_shared_mutex.pml).
- The gate's generation wrap is priced at days of churn; at a modulus of 2 it certifies a probe a mutator ran through: [`basic_shared_mutex.pml`](basic_shared_mutex.pml).
- A certified search's acquire fence before its exit probe pairs with `lock_shared`'s release fence, and `unlock_shared`'s release and `try_lock`'s acquire carry the shared side's edits to the unique side, or a row is read torn: [`basic_shared_mutex.pml`](basic_shared_mutex.pml).
- The basic index's rows carry their edits on the row lock's acquire and the unlock's release, and the traversal admits only IDs below its snapshot: [`basic_ram_index_rows.pml`](basic_ram_index_rows.pml).
- `basic::ram_index::insert` dispenses a batch with one add and publishes it with one turn, and a walk admits another batch's row only once it acquires that row's `wired_k` bit, or a reader sees half a batch or a row unwritten: [`basic_ram_index_batches/serial.pml`](basic_ram_index_batches/serial.pml).
- A planned basic batch's steps release their counts to the `landed` tally that `finish` acquires, `finish` refuses until every position landed, and a refused fan-out lands on the caller, or a reader sees a row unwritten or every later turn waits forever: [`basic_ram_index_batches/planned.pml`](basic_ram_index_batches/planned.pml).
- `back_link_` revalidates the relocked row slot by slot, or an `unlink` followed by an append brings back the unlinked edge, and it runs before its batch's turn, where no `link` can name the newcomer: [`basic_ram_index_back_link.pml`](basic_ram_index_back_link.pml).
- The load path's `finish` stores the counters before the watermark's release, and the steps' tally releases for the finisher's acquire: [`basic_ram_index_steps.pml`](basic_ram_index_steps.pml).
- A failed load wipes the keys it registered, and `insert` clears its row before wiring: [`test/basic/sentences.cpp`](../test/basic/sentences.cpp).
- `chunked_multimap` inserts only once the claim covered the seat's chunk, publishes a slot with a release and probes it with an acquire: [`ram_tables.pml`](ram_tables.pml); and a claimer zeroes its fresh chunk before the compare-exchange that publishes it: [`ram_tables_growth.pml`](ram_tables_growth.pml).
- A `concurrent_k` save copies each row under its slot-zero lock, since a refinement relocks published rows and moves no counter the save re-reads, or it settles a file holding a locked or torn row; it reads a closed dispenser as settled: [`dense_ram_index_watermark/refining.pml`](dense_ram_index_watermark/refining.pml).
- A load or a refinement closes `next_dispensed_id_` by one compare-exchange from the watermark, refused while a batch is between its dispense and its turn, and every dispense refuses while it is closed, or an insert dispenses an ID a load is landing, or appends a back-link that a refinement's rewrite from its staged copy drops: [`dense_ram_index_watermark/closing.pml`](dense_ram_index_watermark/closing.pml).
- A cursor's `seek` refuses an ID at or past its snapshot, and `refresh` acquires the watermark, or a followed back-link locates a row whose chunk has not reached the cursor: [`dense_ram_index_watermark/cursor.pml`](dense_ram_index_watermark/cursor.pml).
- The load path's `finish` moves an open dispenser by one compare-exchange from its base before the watermark's release, which nothing posted can outrun, and leaves a closed one closed: [`dense_ram_index_watermark/appender.pml`](dense_ram_index_watermark/appender.pml).
- A batch dispenses its range with one read-modify-write and takes one watermark turn for it, and a walk admits a row past its snapshot only by an acquire of its slot zero, or it reads that row's payload stale: [`dense_ram_index_batches.pml`](dense_ram_index_batches.pml).
- A turn's waiters re-read the watermark after arming, and the store that ends a turn wakes them, or a batch waits forever for a turn that already came: [`dense_ram_index_batches.pml`](dense_ram_index_batches.pml).
- `release_capacity`'s subtract releases, since a relaxed one is posted under far memory and a claimer takes rows past the only chunk: [`dense_ram_index_capacity/`](dense_ram_index_capacity/).
- A release lands only on a whole counter and the grower's take releases, or two reservations elect two growers: [`dense_ram_index_capacity/reservers.pml`](dense_ram_index_capacity/reservers.pml).
- A second `reserve` above the physical rows claims up to its request, or it spins forever, and a dead load plan releases the rows it claimed: [`test/dense.cuh`](../test/dense.cuh).
- A planned batch claims every shard before it dispenses, claims die only before the dispense, and a refused fan-out lands its rows on the caller, or a turn waits forever on rows nobody dispenses: [`dense_ram_indexes_claims.pml`](dense_ram_indexes_claims.pml).
- Every claimer covers the shard's key column before dispensing, and the dispenser's acquire-release carries a peer's cover along, even when each claimer covers only to its own claim's end, as `basic::ram_index` does, and a refused cover gives its claim back: [`dense_ram_indexes_cover.pml`](dense_ram_indexes_cover.pml).
- The monolith save holds every batch on all of its shards or on none, since a batch dispenses on every shard before it publishes on any, and the load path lands keys before `finish`: [`dense_ram_indexes_plans/`](dense_ram_indexes_plans/).
- A batch across shards stores its watermarks behind a counted sequence word, which a parity word cannot replace for two commits, and dispenses its ranges under one mutex, or two commits deadlock: [`dense_ram_indexes_commits.pml`](dense_ram_indexes_commits.pml).
- The sharded index closes its shards' dispensers in placement order and reopens them the last first, so the closed shards always form a prefix and a batch refused on a closed shard dispensed nothing in that step; closing out of order or reopening in placement order strands a dispensed range: [`dense_ram_indexes_commits.pml`](dense_ram_indexes_commits.pml).
- The CUDA index's host side claims by one compare-exchange from the watermark, so one batch at a time is in flight and grows the storage with no mutex, gives back the claim of a rejected launch and of a load that dies, turns the watermark after the drain, and filters the walk on it, or two batches grow at once, the claim stays above the watermark for good, or a search reads a row in flight: [`dense_vram_index_claims.pml`](dense_vram_index_claims.pml).
- The CUDA rows need a `__syncwarp()` after each shuffle, acquire loads on the tail slots, one release fence per lock hold, and the slot-zero release that carries a newcomer's payload to a peer drawing it as a landmark: [`dense_vram_index_locks.pml`](dense_vram_index_locks.pml).
- The sharded CUDA index claims every shard's share in shard order, closes every shard around its load and its save, writes keys before the turn and the load's `finish`, and commits its turns behind the sequence word, or a load lands on a batch's row and a save copies half a batch: [`dense_vram_indexes/`](dense_vram_indexes/).
- A refine's merge stores every slot of a live row relaxed and never the sentinel into slot zero, so a walk reads each slot old or new, and the closed dispenser keeps a batch's back-links out of the rows it rewrites, or the merge drops a newcomer's only in-link: [`dense_vram_index_merge.pml`](dense_vram_index_merge.pml).
- The NVMe rings' tails release and acquire exactly as the io_uring documentation asks, and `abandon_` waits out every read the kernel consumed before it drops their completions, or a read of the abandoned batch is still in flight when `submit` returns: [`nvme_ring.pml`](nvme_ring.pml).

## What Is Not Covered

The dense index header compiles and starts under GenMC, but its first chunk is a thousand rows of sentinel stores, more events than GenMC's execution graph holds, so the dense clients spell the counters' lines rather than driving the header.
The far-memory results are Spin's alone: GenMC has no posted add.
`chunked_multimap` crashed GenMC 0.18's interpreter before the first insert, single-threaded, so the multimap has no client, and 0.19 is untried.
The rows model has no client either: the row lock sits inside `basic::ram_index`, whose allocations and key map are more than GenMC's graph takes.
The CUDA rows model folds the prior lock holder into one lane and takes `bar.warp.sync`'s ordering among its participants from the ISA text.
A clear, and an absorb consuming its donor, rebuild shards that an insert may still be claiming capacity on before its dispense, which no closed dispenser refuses, so an insert racing either stays outside the contract.
The executors' holds are plain try-locks around the per-thread arenas and are not modelled; the fan-out under them is ForkUnion's, which refuses a second dispatch on its own.
IMM may admit more than real Arm and POWER chips do, so an IMM failure says compiled code is not guaranteed safe there, not that silicon showed it.
Nothing here runs on silicon; litmus tests for the instruction-set references belong beside ForkUnion's probes, and are the next layer down.

## Running

```sh
./check.sh                                       # everything; GenMC is skipped when absent
./check.sh dense_ram_index_watermark/keyed.pml   # one scenario
GENMC=~/genmc/build/bin/genmc CPLUS_INCLUDE_PATH=../forkunion/include ./check.sh
```

The dense clients spell their counters over `fu::standard_atomic_ref`, the reference the index binds, so GenMC needs ForkUnion's headers on `CPLUS_INCLUDE_PATH`: the submodule's, as CI passes them, or any ForkUnion checkout's.
Verdicts run `VERIFY_JOBS` at a time, four by default, since each pan holds a hash table of its own.
