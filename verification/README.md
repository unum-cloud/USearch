# Verification

Model checking for the index protocols: the gates, the watermarks, the capacity counters, the public calls interleaved, the CUDA index's rows and the NVMe index's rings.
[Spin](https://spinroot.com) checks each protocol as a Promela model, and [GenMC](https://github.com/MPI-SWS/genmc) checks the C++ against the real header where it fits, under SC, RC11 and IMM, the model of compiled code on x86, Arm and POWER.

- `weak_memory.pml` is the C++ memory model as views, for Promela, under three models chosen by `-Dmemory=`: `sc`, one copy of every word; `rc11`, the default, RC11's release-acquire-relaxed fragment; and `far`, which also posts a relaxed no-return add the way RAO-INT does.
- `check.sh` is the runner, and `genmc.hpp` the threads and the assertion a client takes from GenMC.
- `watermark_turn.pml` and `basic_gate.pml` are protocols several components include.

## Writing a model

A component is one mechanism of the code under test: a flat `<name>.pml` while it has one scenario, and a directory `<name>/` once it has a second, holding a `protocol.pml` with its words, values, knobs and inlines, one file per scenario, and a `client.cpp` when GenMC checks the same words.
A scenario defines its shape, `thread_count`, `location_count` and `history_depth`, includes its protocol, and starts its roles from one `init { atomic { run <role>(<thread>); … } }`; the depth is the exact minimum every `rc11` and `far` line of the scenario accepts.
There is no conditional compilation: a knob is a choice the header makes, defaulted under `#ifndef` and read as the guard of a plain `if` with an `else` branch, and memory-model differences are expressions on `memory`.
A client spells the same knobs as fields of a `knobs_t`, one `constexpr knobs_t <variant>_k` per variant behind a `template <knobs_t const &knobs_>`, and one `extern "C"` entry per variant, `<scenario>` or `<scenario>_<knob>_<value>`.

Every expected verdict is a `@verify` line in the docblock of the file it checks:

```
@verify <pass|fail|stuck> <memory>[,<memory>...] [<knob>=<value> ...][: <finding>]   in a scenario
@verify <pass|fail|stuck> <memory>[,<memory>...] <entry>[: <finding>]                in a client
```

`pass` is a search that completed with no error, `fail` a violated assertion, and `stuck` an invalid end state in Spin or a liveness violation in GenMC; a search cut short, a knob defined twice, or a write past the scenario's shape is `broken`.
Every scenario has a `pass` line, every knob appears on some line, and adding a scenario, a knob or a finding never edits `check.sh`.

## What the models found

Each finding is the `@verify` line that replays it, where the full counterexample is written out.

- `basic_shared_mutex::lock_shared` ends in a release fence; without it, a certified search certifies a torn pair of rows, under RC11 and under IMM on Arm and POWER, though x86's locked add orders the rows on its own: [`basic_shared_mutex.pml`](basic_shared_mutex.pml), [`basic_shared_mutex.cpp`](basic_shared_mutex.cpp).
- The gate's back-off wait and the deficit wait in `claim_capacity` are capped; uncapped, a mutator arms its monitor after the holder left and waits forever: [`basic_shared_mutex.pml`](basic_shared_mutex.pml).
- The gate's generation wrap is priced at days of churn; at a modulus of 2 it certifies a probe a mutator ran through: [`basic_shared_mutex.pml`](basic_shared_mutex.pml).
- The basic index's rows carry their edits on the row lock's acquire, the unlock's release and `unlock_shared`'s release, and the traversal admits only IDs below its snapshot: [`basic_ram_index_rows.pml`](basic_ram_index_rows.pml).
- The load path's `finish` stores the counters before the watermark's release, and the steps' tally releases for the finisher's acquire: [`basic_ram_index_steps.pml`](basic_ram_index_steps.pml).
- A failed load wipes the keys it registered, `insert` clears its row before wiring, and a successful `finish` spends the plan: [`basic_ram_index.pml`](basic_ram_index.pml).
- `chunked_multimap` inserts only once the claim covered the seat's chunk, publishes a slot with a release and probes it with an acquire: [`ram_tables.pml`](ram_tables.pml); and a claimer zeroes its fresh chunk before the compare-exchange that publishes it: [`ram_tables_growth.pml`](ram_tables_growth.pml).
- The dense index's back-link fence carries a row's relaxed tail stores into a concurrent save's re-check, under RC11 and under IMM: [`dense_ram_index_watermark/`](dense_ram_index_watermark/).
- The load path's identity add releases, since a relaxed one is posted under far memory and lands after the watermark: [`dense_ram_index_watermark/appender.pml`](dense_ram_index_watermark/appender.pml).
- A batch dispenses its range with one add and takes one watermark turn for it, and a walk admits a sibling of its own range only once slot zero shows it published: [`dense_ram_index_batches.pml`](dense_ram_index_batches.pml).
- `release_capacity`'s subtract releases, since a relaxed one is posted under far memory and a claimer takes rows past the only chunk: [`dense_ram_index_capacity/`](dense_ram_index_capacity/).
- A release lands only on a whole counter and the grower's take releases, or two reservations elect two growers: [`dense_ram_index_capacity/reservers.pml`](dense_ram_index_capacity/reservers.pml).
- A second `reserve` above the physical rows claims up to its request, or it spins forever: [`dense_ram_index_capacity/regrow.pml`](dense_ram_index_capacity/regrow.pml).
- A dead load plan releases the rows it claimed: [`dense_ram_index_capacity/loaders.pml`](dense_ram_index_capacity/loaders.pml).
- A shard's claim is the only claim a row makes, released after a refused fan-out and consumed on success, and keys land before the watermark turn: [`dense_ram_indexes_claims.pml`](dense_ram_indexes_claims.pml).
- Every claimer covers the shard's key column before dispensing, and the dispenser's acquire-release carries a peer's cover along: [`dense_ram_indexes_cover.pml`](dense_ram_indexes_cover.pml).
- The monolith save takes one watermark per shard, which never captures a batch across two shards, and the load path lands keys before `finish`: [`dense_ram_indexes_plans/`](dense_ram_indexes_plans/).
- A batch across shards stores its watermarks behind a counted sequence word, which a parity word cannot replace for two commits, and dispenses its ranges under one mutex, or two commits deadlock: [`dense_ram_indexes_commits.pml`](dense_ram_indexes_commits.pml).
- The CUDA index's host side claims under its mutex, reads `plan_load`'s base under the claim, turns the watermark after the drain, filters the walk on it, and unclaims a rejected launch: [`dense_vram_index_claims.pml`](dense_vram_index_claims.pml).
- The CUDA rows need a `__syncwarp()` after each shuffle, acquire loads on the tail slots, and one release fence per lock hold: [`dense_vram_index_locks.pml`](dense_vram_index_locks.pml).
- The sharded CUDA index gates its load as well as its batches, writes keys before the turn and the load's `finish`, and commits its turns behind the sequence word: [`dense_vram_indexes/`](dense_vram_indexes/).
- The NVMe rings' tails release and acquire exactly as the io_uring documentation asks: [`nvme_ring/`](nvme_ring/).
- The executor's hold keeps a second caller from rewinding the arenas under a fan-out: [`executor.pml`](executor.pml).

## What is not covered

The dense index header compiles and starts under GenMC, but its first chunk is a thousand rows of sentinel stores, more events than GenMC's execution graph holds, so the dense clients spell the counters' lines rather than driving the header.
The far-memory results are Spin's alone: GenMC has no posted add.
`chunked_multimap` crashed GenMC 0.18's interpreter before the first insert, single-threaded, so the multimap has no client, and 0.19 is untried.
The rows model has no client either: the row mutex sits inside `basic_ram_index`, whose allocations and key map are more than GenMC's graph takes.
The CUDA rows model folds the prior lock holder into one lane and takes `bar.warp.sync`'s ordering among its participants from the ISA text.
An add overlapping a load of the RAM sharded index is outside that index's contract, which has no mutation gate; the CUDA twin's gate covers its load.
The CUDA executor's hold has the same shape as the RAM one's and is not modelled separately.
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
