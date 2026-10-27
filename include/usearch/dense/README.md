# Dense Indexes

Everything dense, one level deep: equidimensional vectors whose most common similarity measures - Inner-Product, Angular, Euclidean, Hamming, Jaccard - derive from a single inner product of two vectors, so pre-computed per-vector norms cut the arithmetic intensity of the composite metrics by 3×.
The RAM residency is the portable default every platform owns; the VRAM and NVMe siblings stay opt-in through their own extensions, so neither belongs in a portable include.

## Residency

- `ram_metric.hpp` + `ram_index.hpp` - the CPU baseline: compile-time-dispatched NumKong kernel kits chained per capability tier, and the concurrent, watermark-published in-memory index over them.
- `vram_index.hpp` - the host half the NVIDIA and AMD siblings share: the kernels' argument records and the machinery that fills them, claims rows, publishes them and moves checkpoints.
- `vram_nvidia_index.hpp` - the NVIDIA sibling, plain C++ over the kernels of `vram_nvidia_index.cuh` and the per-generation device kits of `vram_nvidia_metric.cuh` - `ampere` through `blackwellrtx` ceilings - under one index whose geometry is frozen at the first `reserve`.
- `vram_amd_index.hpp` - the AMD sibling, plain C++ over the kernels of `vram_amd_index.hip` and the per-generation device kits of `vram_amd_metric.hip` - the ROCm floor through `cdna5` ceilings - under one index whose geometry is frozen at the first `reserve`.
- `vram_index.cuh` - the vendor-neutral device bodies, walk, insert and refine, NVCC and HIP-Clang share, templated on each vendor's warp policy from `utilities/vram_queue.cuh`.
- `vram_apple_index.hpp` - the Apple sibling over unified memory: the same verbs, with the walk and insert kernels in `vram_apple_index.metal` and the kits in `vram_apple_metric.metal`, compiled from embedded source at first use.
- `nvme_index.hpp` - the persisted sibling: one SafeGraphs part walked as a pure state machine of reads, which the engines of `utilities/nvme_device.hpp` submit, from blocking reads through Linux AIO and `io_uring` to SPDK.

## The Plural Layer

The plural follows the sibling idiom - `ram_indexes.hpp` reads as "many `ram_index` viewed as one":

- `utilities/ram_shards.hpp` - the shard table every plural but the NVMe one owns, basic ones included: shards each tower builds, reserves and shape-checks, their key columns and locator map, rotation-balanced routing, and the commit word a reader spanning the shards snapshots by.
- `shard_checkpoints.hpp` - the SafeGraphs save and load plans those plurals open over their shard set, one monolith or a manifest of parts, with what differs per residency handed in as a policy object.
- `utilities/vram_shards.hpp` - `utilities::vram_shard_fanout`, one batched read over NVIDIA or AMD shards: the executor's claim, one context per shard, one snapshot of the visible rows, and the host-landed merge.
- `ram_indexes.hpp` - typed multi-shard views with compute-domain routing and merged cross-shard search, one shard per memory domain.
- `vram_nvidia_indexes.hpp` and `vram_amd_indexes.hpp` - typed multi-shard groups with round-robin ingestion and merged cross-shard search, any number of shards per device.
- `vram_apple_indexes.hpp` - the Apple twin over its one device, so one shard.
- `nvme_indexes.hpp` - read-only views of the parts a RAM or VRAM group saved, every row's walks over all parts on one executor thread, and key lookups as bulk scans of the key columns.
- `nvme_compaction.hpp` - the freestanding `compact`, `refine` and `build` that write new parts from old ones in phases of steps, within a workspace the caller lends, each target part holding at most a shard capacity of rows.

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

Every layer and residency reports one `dense::refine_result_t`, whose counts mean the same everywhere.
The plural layer fans the passes over one executor.
The singular CPU layer hands out plans instead - `plan_refine` and `plan_health` cut each phase into disjoint ID ranges any number of threads may run, with `advance` the one barrier between phases - since one index knows nothing about pools.
