#!/usr/bin/env bash
# Runs every Promela model under the memory models it is meant for, and compares Spin's verdict
# with the expected one: `pass` means no assertion fails, `fail` means the model admits the
# outcome the assertion forbids - the variants that drop a fence or shrink a modulus. The memory
# models and the runner's functions live in ForkUnion's `verification/`.
# Needs `spin` and a C compiler; a GenMC on the path, or named by `GENMC`, also runs the clients.

set -u
cd "$(dirname "$0")"
source ../forkunion/verification/check.sh

section "basic_shared_mutex.pml: the gate, its certified probes, the wrap and the missing release"
verify basic_shared_mutex.pml pass -Dmemory=sequential
verify basic_shared_mutex.pml fail -Dmemory=sequential -Dgeneration_modulus=2
verify basic_shared_mutex.pml pass
verify basic_shared_mutex.pml fail -Dwithout_release_fence
verify basic_shared_mutex.pml fail -Dmemory=sequential -Dwithout_backoff_cap

section "dense_ram_index_watermark.pml: dispensing, publication, the settlement probe, the appender, the keys, the cursor, the load fan-out and two savers"
verify dense_ram_index_watermark.pml pass -Dmemory=sequential
verify dense_ram_index_watermark.pml pass
verify dense_ram_index_watermark.pml fail -Dwithout_watermark_release
verify dense_ram_index_watermark.pml fail -Dwithout_release_fence
verify dense_ram_index_watermark.pml pass -Dmemory=sequential -Dscenario=appender
verify dense_ram_index_watermark.pml pass -Dscenario=appender
verify dense_ram_index_watermark.pml pass -Dmemory=far -Dscenario=appender
verify dense_ram_index_watermark.pml fail -Dmemory=far -Dscenario=appender -Dwithout_appender_release
verify dense_ram_index_watermark.pml fail -Dmemory=sequential -Dscenario=appender -Dtwice_finished
verify dense_ram_index_watermark.pml pass -Dmemory=sequential -Dscenario=keyed
verify dense_ram_index_watermark.pml pass -Dscenario=keyed
verify dense_ram_index_watermark.pml fail -Dmemory=sequential -Dscenario=keyed -Dwithout_key_before_turn
verify dense_ram_index_watermark.pml fail -Dmemory=sequential -Dscenario=keyed -Dwithout_watermark_filter
verify dense_ram_index_watermark.pml pass -Dmemory=sequential -Dscenario=cursor
verify dense_ram_index_watermark.pml pass -Dscenario=cursor
verify dense_ram_index_watermark.pml fail -Dscenario=cursor -Dwithout_watermark_release
verify dense_ram_index_watermark.pml pass -Dmemory=sequential -Dscenario=regions
verify dense_ram_index_watermark.pml pass -Dscenario=regions
verify dense_ram_index_watermark.pml pass -Dmemory=far -Dscenario=regions
verify dense_ram_index_watermark.pml fail -Dscenario=regions -Dwithout_covered_release
verify dense_ram_index_watermark.pml pass -Dmemory=sequential -Dscenario=savers
verify dense_ram_index_watermark.pml pass -Dscenario=savers
verify dense_ram_index_watermark.pml fail -Dscenario=savers -Dwithout_release_fence

section "dense_vram_index_claims.pml: the host's claim under its mutex, the planes' growth, size() as a search bound, and the device's admission ramp"
verify dense_vram_index_claims.pml pass -Dmemory=sequential
verify dense_vram_index_claims.pml pass
verify dense_vram_index_claims.pml fail -Dwithout_claim_release
verify dense_vram_index_claims.pml fail -Dwithout_size_acquire
verify dense_vram_index_claims.pml fail -Dmemory=sequential -Dwithout_claiming_mutex
verify dense_vram_index_claims.pml fail -Dmemory=sequential -Dwithout_unclaim_on_rejection
verify dense_vram_index_claims.pml fail -Dmemory=sequential -Dwithout_base_under_claim
verify dense_vram_index_claims.pml fail -Dmemory=sequential -Dwithout_abandon_unlock

section "dense_vram_indexes.pml: the mutation gate, the ceiling, keys before the claim, and a load beside a batch"
verify dense_vram_indexes.pml pass -Dmemory=sequential
verify dense_vram_indexes.pml pass
verify dense_vram_indexes.pml fail -Dmemory=sequential -Dwithout_gate
verify dense_vram_indexes.pml fail -Dmemory=sequential -Dwithout_key_before_claim
verify dense_vram_indexes.pml fail -Dwithout_key_before_claim
verify dense_vram_indexes.pml fail -Dwithout_claim_release
verify dense_vram_indexes.pml pass -Dmemory=sequential -Dscenario=loading
verify dense_vram_indexes.pml pass -Dscenario=loading
verify dense_vram_indexes.pml fail -Dmemory=sequential -Dscenario=loading -Dwithout_load_gate
verify dense_vram_indexes.pml fail -Dmemory=sequential -Dscenario=loading -Dwithout_empty_under_gate
verify dense_vram_indexes.pml fail -Dmemory=sequential -Dscenario=loading -Dwithout_load_key_before_finish

section "executor.pml: the one-caller hold, the arena rewind, and the fan-out"
verify executor.pml pass -Dmemory=sequential
verify executor.pml pass
verify executor.pml fail -Dmemory=sequential -Dwithout_hold

section "nvme_ring.pml: the io_uring words, the wrap, and the deferred posting"
verify nvme_ring.pml pass -Dmemory=sequential
verify nvme_ring.pml pass
verify nvme_ring.pml fail -Dwithout_sq_tail_release
verify nvme_ring.pml fail -Dwithout_cq_tail_acquire
verify nvme_ring.pml pass -Dscenario=deferred
verify nvme_ring.pml fail -Dscenario=deferred -Dwithout_cq_tail_acquire

section "dense_vram_index_locks.pml: the CUDA rows, their slot-zero locks, and what a warp's follower lanes see"
verify dense_vram_index_locks.pml pass -Dmemory=sequential
verify dense_vram_index_locks.pml pass -Dmemory=sequential -Dwithout_fixes
verify dense_vram_index_locks.pml pass
verify dense_vram_index_locks.pml fail -Dwithout_syncwarp_after_broadcast
verify dense_vram_index_locks.pml fail -Dwithout_syncwarp_after_lock
verify dense_vram_index_locks.pml fail -Dwithout_tail_acquire
verify dense_vram_index_locks.pml fail -Dwithout_tail_fence
verify dense_vram_index_locks.pml fail -Dwithout_fixes

section "dense_ram_index_capacity.pml: the chunk election and its flag, the grower's take, release_capacity's order, a second reservation above the physical rows, two reservations, and an abandoned load plan"
verify dense_ram_index_capacity.pml pass -Dmemory=sequential
verify dense_ram_index_capacity.pml pass
verify dense_ram_index_capacity.pml pass -Dmemory=far
verify dense_ram_index_capacity.pml fail -Dmemory=far -Dwithout_deflation_release
verify dense_ram_index_capacity.pml pass -Dmemory=sequential -Dscenario=regrow
verify dense_ram_index_capacity.pml pass -Dscenario=regrow
verify dense_ram_index_capacity.pml pass -Dmemory=far -Dscenario=regrow
verify dense_ram_index_capacity.pml fail -Dmemory=sequential -Dscenario=regrow -Dwithout_deficit_growth
verify dense_ram_index_capacity.pml pass -Dmemory=sequential -Dscenario=reservers
verify dense_ram_index_capacity.pml pass -Dscenario=reservers
verify dense_ram_index_capacity.pml pass -Dmemory=far -Dscenario=reservers
verify dense_ram_index_capacity.pml pass -Dmemory=sequential -Dscenario=reservers -Dsame_geometry
verify dense_ram_index_capacity.pml fail -Dmemory=far -Dscenario=reservers -Dwithout_deflation_release
verify dense_ram_index_capacity.pml fail -Dmemory=sequential -Dscenario=reservers -Dwithout_whole_credit
verify dense_ram_index_capacity.pml fail -Dscenario=reservers -Dwithout_take_release
verify dense_ram_index_capacity.pml pass -Dmemory=sequential -Dscenario=loaders
verify dense_ram_index_capacity.pml pass -Dscenario=loaders
verify dense_ram_index_capacity.pml pass -Dmemory=far -Dscenario=loaders
verify dense_ram_index_capacity.pml fail -Dmemory=sequential -Dscenario=loaders -Dwithout_abandon_release

section "dense_ram_indexes.pml: the shards' claims, keys before the turn, the monolith cut per shard, and the load path's keys"
verify dense_ram_indexes.pml pass -Dmemory=sequential
verify dense_ram_indexes.pml pass
verify dense_ram_indexes.pml fail -Dmemory=sequential -Dwithout_key_before_watermark
verify dense_ram_indexes.pml fail -Dwithout_key_before_watermark
verify dense_ram_indexes.pml fail -Dwithout_watermark_release
verify dense_ram_indexes.pml fail -Dmemory=sequential -Dwithout_release
verify dense_ram_indexes.pml fail -Dmemory=sequential -Dwithout_claim_consume
verify dense_ram_indexes.pml pass -Dmemory=sequential -Dscenario=saving
verify dense_ram_indexes.pml pass -Dscenario=saving
verify dense_ram_indexes.pml fail -Dmemory=sequential -Dscenario=saving -Dwhole_batches
verify dense_ram_indexes.pml fail -Dscenario=saving -Dwhole_batches
verify dense_ram_indexes.pml pass -Dmemory=sequential -Dscenario=loading
verify dense_ram_indexes.pml pass -Dscenario=loading
verify dense_ram_indexes.pml pass -Dmemory=far -Dscenario=loading
verify dense_ram_indexes.pml fail -Dmemory=sequential -Dscenario=loading -Dwithout_keys_before_finish
verify dense_ram_indexes.pml fail -Dscenario=loading -Dwithout_keys_before_finish

section "basic_ram_index_rows.pml: the rows under their mutex, what the gate carries, the traversal's frontier, and the load path's regions"
verify basic_ram_index_rows.pml pass -Dmemory=sequential
verify basic_ram_index_rows.pml pass
verify basic_ram_index_rows.pml fail -Dwithout_watermark_release
verify basic_ram_index_rows.pml fail -Dwithout_row_lock_acquire
verify basic_ram_index_rows.pml fail -Dwithout_row_unlock_release
verify basic_ram_index_rows.pml fail -Dwithout_unlock_shared_release
verify basic_ram_index_rows.pml fail -Dmemory=sequential -Dwithout_watermark_filter
verify basic_ram_index_rows.pml pass -Dmemory=sequential -Dscenario=regions
verify basic_ram_index_rows.pml pass -Dscenario=regions
verify basic_ram_index_rows.pml pass -Dmemory=far -Dscenario=regions
verify basic_ram_index_rows.pml fail -Dscenario=regions -Dwithout_covered_release
verify basic_ram_index_rows.pml fail -Dmemory=far -Dscenario=regions -Dwithout_covered_release
verify basic_ram_index_rows.pml fail -Dscenario=regions -Dwithout_counters_before_watermark

section "basic_ram_index.pml: add, search, save, health and load interleaved on one index, and a failed load's leftovers"
verify basic_ram_index.pml pass
verify basic_ram_index.pml fail -Dwithout_row_clear
verify basic_ram_index.pml fail -Dwithout_map_clear
verify basic_ram_index.pml fail -Dtwice_finished

section "ram_tables.pml: the multimap's quota, its slot claim, and what a finder sees beside a published state"
verify ram_tables.pml pass -Dmemory=sequential
verify ram_tables.pml pass
verify ram_tables.pml pass -Dmemory=far
verify ram_tables.pml fail -Dwithout_publish_release
verify ram_tables.pml fail -Dwithout_probe_acquire
verify ram_tables.pml fail -Dmemory=sequential -Dwithout_bounded_quota
verify ram_tables.pml fail -Dwithout_bounded_quota

section "GenMC clients over the real headers"
if genmc_ready; then
    verify_client basic_shared_mutex.cpp pass -I../include -I../forkunion/include
    verify_client dense_ram_index_watermark.cpp pass -I../forkunion/include
    verify_client dense_ram_index_watermark.cpp fail -I../forkunion/include -Dwithout_release_fence
    verify_client dense_ram_index_capacity.cpp pass -I../forkunion/include
    verify_client dense_ram_index_capacity.cpp fail -I../forkunion/include -Dwithout_credit_release
fi

finish
