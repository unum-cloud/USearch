# Dense Indexes

Everything dense, one level deep: equidimensional vectors whose most common similarity measures - Inner-Product, Angular, Euclidean, Hamming, Jaccard - derive from a single inner product of two vectors, so pre-computed per-vector norms cut the arithmetic intensity of the composite metrics by 3x.
The RAM residency is the portable default every platform owns; the VRAM and NVMe siblings stay opt-in through their own extensions, so neither belongs in a portable include.

## Residencies

- `ram_metric.hpp` + `ram_index.hpp` - the CPU baseline: compile-time-dispatched NumKong kernel kits chained per capability tier, and the concurrent, watermark-published in-memory index over them.
- `vram_metric.cuh` + `vram_index.cuh` - the CUDA sibling: per-generation device kits - `ampere` through `blackwell` ceilings - under one index whose geometry is frozen at the first `reserve`.
- `nvme_index.hpp` - the persisted sibling: a pure traversal state machine pumped by pluggable I/O engines, from the portable POSIX floor through `io_uring` rings to NVMe passthrough; its engine kits ride here until a second modality needs them.

## The Plural Layer

The plural follows the sibling idiom - `ram_indexes.hpp` reads as "many `ram_index` viewed as one":

- `ram_indexes.hpp` - typed multi-shard views with compute-domain routing and merged cross-shard search, one shard per memory domain.
- `vram_indexes.cuh` - typed multi-shard groups with round-robin ingestion and merged cross-shard search, one shard per device.

Capability files split beside the residency pairs as they grow - `icelake.hpp`, `sapphireamx.hpp`, `hopper.cuh`, and their siblings.

## Construction

Every residency builds through the same three verbs, in this order:

- `insert` wires each node against the graph as it stood, so a narrow `expansion_factor` ingests cheaply and leaves a graph whose picks a wider walk could improve on.
- `refine` re-selects a node range's neighbors against the finished graph, back-linking as an insertion does.
It claims no identity and moves no watermark, so searches run throughout and a stopped sweep merely leaves the graph less improved.
Call it until a pass re-wires nothing; the first two passes carry most of the gain.
- `heal` re-links whatever no search can reach: the audit's islands re-select from scratch, and whatever the diversity judgment still drops is reconnected into a reached pick, round after round while the island population shrinks.

`health` is the read-only half of `heal` - the same landmark sweep, reporting the reachable population and the levels it took, with no edit.
Use it to judge a graph without changing it.

The plural layer fans all four over one executor.
The singular layer hands out plans instead - `plan_health` and `plan_heal` return disjoint ID ranges any number of threads may sweep, with the level and round boundaries left to the caller - since one index knows nothing about pools.
