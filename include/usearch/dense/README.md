# Dense Indexes

Everything dense, one level deep: equidimensional vectors whose most common similarity measures - Inner-Product, Angular, Euclidean, Hamming, Jaccard - derive from a single inner product of two vectors, so pre-computed per-vector norms cut the arithmetic intensity of the composite metrics by 3×.
The RAM residency is the portable default every platform owns; the VRAM and NVMe siblings stay opt-in through their own extensions, so neither belongs in a portable include.

## Residencies

- `ram_metric.hpp` + `ram_index.hpp` - the CPU baseline: compile-time-dispatched NumKong kernel kits chained per capability tier, and the concurrent, watermark-published in-memory index over them.
- `vram_metric.cuh` + `vram_index.cuh` - the CUDA sibling: per-generation device kits - `ampere` through `blackwellrtx` ceilings - under one index whose geometry is frozen at the first `reserve`.
- `nvme_index.hpp` - the persisted sibling: one SafeGraphs part walked as a pure state machine of reads, which the engines of `utilities/nvme_device.hpp` submit, from blocking reads through Linux AIO and `io_uring` to SPDK.

## The Plural Layer

The plural follows the sibling idiom - `ram_indexes.hpp` reads as "many `ram_index` viewed as one":

- `ram_indexes.hpp` - typed multi-shard views with compute-domain routing and merged cross-shard search, one shard per memory domain.
- `vram_indexes.cuh` - typed multi-shard groups with round-robin ingestion and merged cross-shard search, one shard per device.
- `nvme_indexes.hpp` - read-only views of the parts a RAM or VRAM group saved, every row's walks over all parts on one executor thread, and key lookups as bulk scans of the key columns; beside them, the freestanding `merge`, `split`, `refine` and `build` that write new parts from old ones in phases of steps, within a workspace the caller lends.

Capability files split beside the residency pairs as they grow - `icelake.hpp`, `sapphireamx.hpp`, `hopper.cuh`, and their siblings.

## Construction

Every residency builds through the same two verbs, in this order:

- `insert` wires each node against the graph as it stood, so a narrow `expansion_factor` ingests cheaply and leaves a graph whose picks a wider walk could improve on.
- `refine` re-selects a node range's neighbors against the finished graph, back-linking as an insertion does, then heals whatever no search can reach: the audit's islands re-select from scratch, and whatever the diversity judgment still drops is reconnected into a reached pick, round after round while the island population shrinks.
  It claims no identity and moves no watermark, so a stopped pass merely leaves the graph less improved.
  Call it until a pass re-wires nothing; the first two passes carry most of the gain.
  A `refine` over no nodes is the heal alone, which the plural layer names `heal`.

`health` is the read-only audit the heal starts from - the same landmark sweep, reporting the reachable population and the levels it took, with no edit.
Use it to judge a graph without changing it.

Every layer and residency reports one `dense_refine_result_t`, whose counts mean the same everywhere.
The plural layer fans the passes over one executor.
The singular CPU layer hands out plans instead - `plan_refine` and `plan_health` cut each phase into disjoint ID ranges any number of threads may run, with `advance` the one barrier between phases - since one index knows nothing about pools.
