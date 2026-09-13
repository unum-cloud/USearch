# Verification

Model checking for the index protocols: the gate, the watermark, the capacity counters, the public calls interleaved, and the CUDA index's rows.
[Spin](https://spinroot.com) checks each protocol as a Promela model under three memory models; [GenMC](https://github.com/MPI-SWS/genmc) checks the C++ under RC11, against the real header where it fits.
The memory models and the runners live in [`forkunion/verification`](../forkunion/verification/README.md); `./check.sh` runs everything here and compares each verdict with the expected one.

- `basic_shared_mutex.pml`: `basic_shared_mutex`, its exclusion, the certified probe bracket, the generation wrap, the capped back-off wait, and whether a certified probe can see a mutator's pair of rows torn.
- `basic_shared_mutex.cpp`: the real header under GenMC, one mutator, one prober, the torn pair.
- `basic_ram_index_rows.pml`: the basic index's rows, the row mutex, what the gate's shared and unique sides carry, the traversal's translation frontier, and the load path's regions, tally and counter order.
- `basic_ram_index.pml`: `insert`, certified `search`, `plan_save`, `health` and `plan_load` interleaved on one index, sequentially consistent, and a failed load's leftovers, the composition of the models above.
- `ram_tables.pml`: `growing_multimap`, the quota bounding every probe chain, the acquire slot claim, and whether a finder can read a published slot half-written.
- `dense_ram_index_watermark.pml`: the dense index's dispenser, rows, back-link fence, watermark turn, and the concurrent save's settlement probe; `-Dscenario=appender` for the load path's `finish`, `keyed` for the key plane and the walk's frontier, `cursor` for `dense_ram_cursor`, `regions` for the load fan-out's tally, `savers` for two saves at once, and `-Dtwice_finished` for a load plan published twice.
- `dense_ram_index_watermark.cpp`: the same counters under GenMC, spelled over the ForkUnion reference the index binds.
- `dense_ram_index_capacity.pml`: `claim_capacity`'s grower election, `release_capacity`'s order and its wait for a whole counter, `-Dscenario=regrow` for a second `reserve` above the physical rows, `reserve` against a concurrent `reserve`, and `-Dscenario=loaders` for a load plan that dies before `finish`.
- `dense_ram_index_capacity.cpp`: the same counters under GenMC, with the race detector on the plain chunk directory.
- `dense_ram_indexes.pml`: the sharded index, the shards' claims and their release, keys before the watermark turn, the search merge over the planes, the monolith save's per-shard cut, and the load path's keys under `finish`.
- `dense_vram_index_claims.pml`: the CUDA index's host side, the claim under its spin mutex, the planes grown before it, `size()` as a search's sampling bound, the load's claim, and the admission ramp behind a rejected launch; `-Dwithout_abandon_unlock` keeps a dead load plan's mutex.
- `dense_vram_index_locks.pml`: the CUDA index's rows, publication by one release, the bounded slot-zero lock, the back-link's sweep and append, and a traversal warp whose follower lanes read behind a shuffle.
- `dense_vram_indexes.pml`: the sharded CUDA index, the mutation gate answering `contended_k`, the admission ceiling, keys written before a shard's claim publishes the row, and a load beside a batch; `-Dscenario=loading` for the load.
- `nvme_ring.pml`: the NVMe index's io_uring words, the submission and completion rings' heads and tails, the entries between them, the wrap, and the deferred posting; `-Dscenario=deferred` for `DEFER_TASKRUN`.
- `executor.pml`: `executor_t`, the one-caller hold through `claim`, the arena rewind in `reserve`, and the fan-out no second caller may rewind under.

## What the models found, and what changed for it

The basic index's certified search was sound on every ISA and unsound in RC11.
`lock_shared` is an acquire read-modify-write, and nothing released between it and the mutator's row writes, so the C++ model let a certified prober read a store made under the lock while its exit probe still read the untouched word: a torn edit certified as quiet.
Spin found it in `basic_shared_mutex.pml`, and GenMC reproduced it on the real header in `basic_shared_mutex.cpp`.
On Arm, x86 and RVWMO an acquire read-modify-write orders the stores after it and the acquire fence orders the reads before the exit probe, which closes the cycle, so hardware never showed it.
`lock_shared` now ends in one `std::atomic_thread_fence(std::memory_order_release)`, the fence the dense index already kept at the head of `back_link_neighbors`; `-Dwithout_release_fence` keeps the counterexample.

`release_capacity`'s load-bearing order did not survive the far mapping.
The relaxed no-return subtract on `last_allocated_id_` preceded the release credit so that the physical-rows sum stays conservative, and under `-Dmemory=far` the posted subtract was still in flight when the credit landed: `dense_ram_index_capacity.pml` showed a claimer taking rows past the end of the only chunk.
The subtract now releases, which no reference posts, and `capacity()` reads the credit before the claim, the order that makes a seen credit carry its deflation; `-Dwithout_deflation_release` keeps the counterexample.
The load path's `finish` had the milder shape, a spurious contention from its relaxed identity add; that add releases now too, with `-Dwithout_appender_release` keeping the old one.
Both far results follow Intel's own rule for the RAO-INT instructions, which the far model transcribes; the hardening costs one lock-prefixed instruction on a cold path either way.

Two waits are correct only because they are capped.
With `-Dwithout_backoff_cap`, a mutator backing off behind a unique holder arms its monitor after the holder already left, and Spin reports the stuck mutator; the same shape governs the deficit wait in `claim_capacity`.
And `-Dgeneration_modulus=2` makes the generation wrap the header prices at days of churn happen in three steps, certifying a probe a mutator ran through.

The CUDA index's follower lanes stand outside their leader's acquire.
`dense_vram_index_locks.pml` gives a warp two lanes with separate views, merged only by `__syncwarp`, and a `__shfl_sync` that moves one value and no view, which is what the PTX ISA promises of each.
Three shapes fail under the view model and pass sequentially.
A traversal's follower lanes load their tail slots straight after the shuffle, so a lane may read a slot older than the unlock its leader acquired and the warp assembles a row no holder ever wrote; `-Dwithout_syncwarp_after_broadcast`.
A back-linker's follower lanes sweep the locked row the same way, so a stale sentinel names a slot the previous holder just filled, and the append overwrites that holder's link; `-Dwithout_syncwarp_after_lock`.
And a follower's relaxed tail load may return an edit newer than the unlock its leader acquired, which carries nothing, so the id it found names a node whose payload it then reads stale; no syncwarp closes that one, since no lane acquired that edit.
The header now carries the model's default: a `__syncwarp()` after each shuffle, acquire loads on the tail slots, and one release fence per lock hold between the lock and the slot stores.
`-Dwithout_tail_acquire` and `-Dwithout_tail_fence` each keep the last counterexample, and `-Dwithout_fixes` is the header as it was, carrying none of the three.

The basic index's rows carry their edits on exactly the orders the header names, and the model says what each one is for.
`basic_ram_index_rows.pml` puts one row under `link`, `unlink` and an insert's `back_link_`, copies it through the row mutex as `copy_row_`, the cursor and `degree_of` do, and reads it plain from the unique side as `health` and `plan_save` do.
Dropping the row lock's acquire or the unlock's release hands the next holder the new size beside an unwritten slot, and dropping `unlock_shared`'s release hands the unique side the same torn row through the gate; `-Dwithout_row_lock_acquire`, `-Dwithout_row_unlock_release` and `-Dwithout_unlock_shared_release` keep the three.
The traversal's translation frontier turned out to be a protocol claim rather than an ordering one: a reader whose snapshot predates a newcomer copies that newcomer's edge from a row it locked, and the lock carries the record, so under the view model following the edge reads the key just fine.
What the filter keeps is that every record a walk reads was published before the walk's snapshot, and `-Dwithout_watermark_filter` breaks that sequentially.
The load path's `finish` has its counters in the right order, the claim and identity stores before the watermark's release, so a reader that acquired the watermark sees the dispenser with it; `-Dwithout_counters_before_watermark` reverses them.
The regions' tally add releases for the finisher's acquire, which is what puts the regions' records behind the watermark; `-Dwithout_covered_release` loses them under the view model and, posted, under far memory.

A failed load left its keys behind.
`load_plan_t::load` registers keys as it decodes, and a plan that failed kept them until the next `plan_load` wiped the map, so an `insert` in between dispensed ids the failed attempt's keys still named and burned quota on the leftovers.
The plan now wipes the keys it registered when it dies before `finish`, whether its regions fell short or nobody finished it, still under its hold, and `insert` clears its row before wiring; `basic_ram_index.pml` carries both as `-Dwithout_map_clear` and `-Dwithout_row_clear`.
A successful `finish` spends the plan, so a second call answers a status instead of dropping a hold the plan no longer owns; `-Dtwice_finished` unlocks twice and the gate's assertion names it, or an adder walks in on a loader's hold.

The multimap's quota counter passed its own quota.
`insert` claimed with an unconditional add and handed the seat back with a subtract on refusal, so between the two a `size()` read one entry past `capacity()`, and one more per refusal in flight; no caller compares the two today, but the docs promise the bound and the counter is the only one the probe chains have.
`ram_tables.pml` gives `size()` its own reader now, and the claim is a compare-exchange that never passes the quota, so the refusal path writes nothing at all; `-Dwithout_bounded_quota` keeps the old claim and the reader sees the overshoot.
The rest of the multimap's protocol holds as written, under every memory model: two admitted inserters never probe past the second slot, and the one order that carries is the pair the header names, the release on the `published_k` store and the acquire on each probe.
`-Dwithout_publish_release` and `-Dwithout_probe_acquire` each let the finder read the new key beside the slot's initial value, since the key and the value are plain stores the state alone publishes, and a `building_k` slot is occupied and never the end of a chain, which is what makes a key published before the finder started reachable through it.

Two reservations broke the grower election.
`claim_capacity` elected the thread whose subtract took `remaining_capacity_` from non-negative to negative and reasoned that the counter cannot recover until that thread credits it, which held only while nobody else credited.
`reserve` claims its deficit and hands it back through `release_capacity`, and `dense_ram_index_capacity.pml` under `-Dscenario=reservers` showed a second reserver elected off the first one's growth credit, the first one's hand-back lifting the counter back to zero while the second still grew, and the first one's next subtract crossing zero again: two threads inside `create_next_chunk`, whose `chunks_count_` and directory writes are plain.

The dense load plan claimed its rows in `plan_load` and released them only through `finish`, so a plan that died in between leaked them: `last_allocated_id_` stayed past the watermark and every later `plan_load` answered `contended_k`.
The plan now owns its claim, whose death releases the rows, and `finish` consumes the claim before publishing and spends the plan; `dense_ram_index_capacity.pml` under `-Dscenario=loaders` carries the leak as `-Dwithout_abandon_release`, and `dense_ram_index_watermark.pml` under `-Dscenario=appender` carries the double publication as `-Dtwice_finished`.
The CUDA load plan held its claim mutex in a public `std::unique_lock` and published through `finish` without spending, so a second `finish` re-stored the claim and unlocked twice; it now spends like its host twin, and `dense_vram_index_claims.pml` lets the loader die before `finish`, with `-Dwithout_abandon_unlock` keeping the mutex it should drop.
A hand-back now credits only a whole counter, waiting out a growth the way a deficit does, so nothing but the grower's own credit lifts a negative counter and no second crossing can happen while one is inside; `-Dwithout_whole_credit` keeps the counterexample.
The same scenario found the grower's take on `last_allocated_id_` acquiring only: a claimer that fit on the old credit but whose rows the grower's take pushed into the new chunk had no edge to the directory, and read it stale.
That take releases now, and `-Dwithout_take_release` keeps the stale read.

The sharded index claims on one shard at a time, and its cut is per shard.
`dense_ram_indexes.pml` runs two batch adders and a single-row adder against two shards of two rows: the shard's counter is the only word a row claims, and once every adder is done each shard's claim word equals its dispenser, unless `-Dwithout_release` skips the release after a refused fan-out.
A key lands in `on_assigned` before the watermark turn, so every id below a watermark a searcher acquired resolves through the plane; `-Dwithout_key_before_watermark` moves it after and fails under every model, `-Dwithout_watermark_release` fails under the view model.
The monolith save takes one `plan_save` per shard, and a settled plan copies no unset key below its cut; `-Dwhole_batches`, which asks the cut to capture a batch's rows on shard zero and shard one together, fails under every model, since nothing joins the two cuts, and the README says so rather than the code.
The load path lands keys before `finish` releases the dispenser and the watermark, and `-Dwithout_keys_before_finish` keeps the counterexample.
The single-row `add` did not consume its claim on success, so every landed row released one seat and the shard claimed one row more than it holds; it consumes now, and `-Dwithout_claim_consume` keeps the old header.
The shards' key planes were a second contended word beside each shard's capacity counter: a `fetch_add_if_at_most` on the plane, then the claim, with two releases to keep aligned.
The shard's claim is now the only claim a row makes: nothing bounds a shard from outside, and the key plane is sized to what the shard really holds after its reservation rounds up to whole chunks.
`reserve` called a second time, above the physical rows, spun forever: a deficit that fits in the free rows is granted without a growth, released, and read back unchanged.
It now claims from the claimed rows up to the request, so the counter crosses and one growth covers it; `dense_ram_index_capacity.pml -Dscenario=regrow` bounds the rounds, and `-Dwithout_deficit_growth` keeps the old deficit and over-runs them.

The CUDA index's host side rests on one mutex and one release.
`dense_vram_index_claims.pml` holds two inserters, a loader, a searcher, and a device serving the accepted batches' rows through the admission ramp in any order.
The claim's release store pairs with the acquire in `size()`, so a search's sampling bound lies inside the planes the claim grew; `-Dwithout_claim_release` and `-Dwithout_size_acquire` each let the searcher read a bound past the planes' extent.
The mutex keeps the growers and the claims apart, and storing the claim only once the launch is accepted keeps a rejected grid from starving the next batch's ramp; `-Dwithout_claiming_mutex` claims one range twice, and `-Dwithout_unclaim_on_rejection` leaves the device waiting for rows nobody publishes.
The model also found `plan_load` reading its base from `size()` before `lock_if_free`, so a batch claimed between the two lines overlapped the loaded rows and `finish` could lower the claim beneath it; the base is read under the claim now, and `-Dwithout_base_under_claim` keeps the old order.
The planes' committed extent, which `capacity()` reads outside the claim, was a plain word the grower advanced under the mutex; it is released by the grower and acquired by every reader now, in both plane allocators.
The far model is not run there: the posted add is x86's, and the ramp's add runs on the GPU.

The sharded CUDA index gated its batches and not its load.
`add` takes the executor's hold and then `mutating_`, so two batches never interleave their claims on `admitted_rows_`; `load_parts_` took only the executor's, so a batch through a second executor passed the gate, its compare-exchange raced the plain total the load stores at its end, and its keys raced the load's.
`dense_vram_indexes.pml` runs two batches, a searcher and a loader over two shards of two rows, and found three things in the load: the missing gate, an emptiness check made at the entry and not under the hold, so a batch admitted between the two was overwritten by the load's total and the next batch's first node reached the plane, and each shard published before its keys were written, the order the RAM twin had already corrected.
The load takes the gate now, checks emptiness under it, and writes each shard's keys before its plan finishes; `-Dwithout_load_gate`, `-Dwithout_empty_under_gate` and `-Dwithout_load_key_before_finish` keep the three.
The relaxed compare-exchange on `admitted_rows_` stays relaxed: every writer is gated, and no reader derives a visibility bound from it, since searches bound on each shard's claim, whose release and acquire `-Dwithout_claim_release` shows to carry the keys.

The NVMe index's ring words carry exactly the orders the io_uring documentation asks for.
`nvme_ring.pml` runs an engine and a kernel over two rings of depth two and three requests, so each ring wraps once: the tail releases and the tail acquires carry the entries, and `-Dwithout_sq_tail_release` lets the kernel consume an entry before its payload landed while `-Dwithout_cq_tail_acquire` lets the engine reap a completion stale.
The acquire on the submission head and the release on the completion head guard the shape where a slot is reused before its reader is done, which RC11 forbids by construction; the view model has no eventual visibility either, so a relaxed head read can stay stale forever and the run ends with a stuck engine rather than a failed assertion.
They stay, since the kernel is not a C++ thread.
Deferred posting changes when completions land and nothing about what is seen.

The executor's hold is what keeps a second caller from rewinding the arenas under a fan-out.
`executor.pml` runs two callers and two workers: `claim` answers `contended_k` rather than waiting, the holder rewinds every arena and fans out, and each worker reads the arena its holder wrote; `-Dwithout_hold` lets the second caller rewind under the first one's fan-out.

## What is not covered

The dense index header compiles and starts under GenMC, but its first chunk is a thousand rows of sentinel stores, more events than GenMC's execution graph holds, so `dense_ram_index_watermark.cpp` and `dense_ram_index_capacity.cpp` spell the counters' lines rather than driving the header.
GenMC knows RC11 only: the far-memory results are Spin's alone.
`growing_multimap` crashes GenMC 0.18's interpreter before the first insert, single-threaded, so `ram_tables.pml` stands alone and the multimap has no client.
The rows model has no client either: the row mutex sits inside `basic_ram_index`, whose allocations and key map are more than GenMC's graph takes.
The CUDA rows model folds the prior lock holder into one lane and takes `bar.warp.sync`'s ordering among its participants from the ISA text.
An add overlapping a load of the RAM sharded index is outside that index's contract, which has no mutation gate; the CUDA twin's gate covers its load now.
The CUDA executor's hold has the same shape as the RAM one's and is not modelled separately.
The traversal's deferral of a locked candidate, proposed beside the four-site fix, waits for the multi-way traversal work.
Nothing here runs on silicon; litmus tests for the instruction-set references belong beside ForkUnion's probes, and are the next layer down.

## Running

```sh
./check.sh                                   # Spin only, GenMC skipped when absent
GENMC=~/genmc/build/bin/genmc ./check.sh     # both
```

Every `verify` line names a model, the expected verdict and the defines, so a new variant is one line.
The suite's 120 verdicts take about four minutes serially and about a minute four at a time, which is the default; a client that runs past `GENMC_SECONDS` is a client to shrink.
