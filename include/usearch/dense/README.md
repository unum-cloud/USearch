# Dense Indexes

Everything dense, one level deep: equidimensional vectors whose most common similarity measures - Inner-Product, Angular, Euclidean, Hamming, Jaccard - derive from a single inner product of two vectors, so pre-computed per-vector norms cut the arithmetic intensity of the composite metrics by 3×.
The RAM residency is the portable default every platform owns; the VRAM and NVMe siblings stay opt-in through their own extensions, so neither belongs in a portable include.

## Residency

- `ram_metric.hpp` + `ram_index.hpp` - the CPU baseline: compile-time-dispatched NumKong kernel kits chained per capability tier, and the concurrent, watermark-published in-memory index over them.
- `vram_index.hpp` - the host half the NVIDIA and AMD siblings share: the kernels' argument records and the machinery that fills them, claims entries, publishes them and moves checkpoints.
- `vram_nvidia_index.hpp` - the NVIDIA sibling, plain C++ over the kernels of `vram_nvidia_index.cuh` and the per-generation device kits of `vram_nvidia_metric.cuh` - `ampere` through `blackwellrtx` ceilings - under one index whose geometry is fixed at construction.
- `vram_amd_index.hpp` - the AMD sibling, plain C++ over the kernels of `vram_amd_index.hip` and the per-generation device kits of `vram_amd_metric.hip` - the ROCm floor through `cdna5` ceilings - under one index whose geometry is fixed at construction.
- `vram_index.cuh` - the vendor-neutral device bodies, walk, insert and refine, NVCC and HIP-Clang share, templated on each vendor's warp policy from `utilities/vram_queue.cuh`.
- `vram_apple_index.hpp` - the Apple sibling over unified memory: the same verbs, with the walk and insert kernels in `vram_apple_index.metal` and the kits in `vram_apple_metric.metal`, compiled from embedded source at first use.
- `nvme_index.hpp` - the persisted sibling: one SafeGraphs part walked as a pure state machine of reads, which the engines of `utilities/nvme_device.hpp` submit, from blocking reads through Linux AIO and `io_uring` to SPDK.

The kits stay in one file per residency, `ram_metric.hpp` and the `vram_*_metric.*` files, since the class units already compile them apart.

## Sharded Indexes

A `*_index` is a mono-index and a `*_indexes` the sharded index over shards of it, so `ram_indexes.hpp` reads as "many `ram_index` viewed as one":

- `utilities/ram_shards.hpp` - the shard table every sharded index but the NVMe one owns, basic ones included: shards each tower builds, reserves and shape-checks, each owning its keys, the reads spanning them, rotation-balanced routing, and the commit word a reader spanning the shards snapshots by.
- `shard_checkpoints.hpp` - the SafeGraphs save and load plans those sharded indexes open over their shard set, one monolith or a manifest of parts, with what differs per residency handed in as a policy object.
- `utilities/vram_shards.hpp` - `utilities::vram_shard_fanout`, one batched read over NVIDIA or AMD shards: the executor's claim, one context per shard, one snapshot of the visible entries, and the await on the first shard's stream, where every shard's rankings merge before the host sleeps once.
- `ram_indexes.hpp` - the sharded index over CPU-resident shards with compute-domain routing and merged cross-shard search, one shard per memory domain.
- `vram_nvidia_indexes.hpp` and `vram_amd_indexes.hpp` - the sharded indexes over NVIDIA and AMD shards with round-robin ingestion and merged cross-shard search, any number of shards per device.
- `vram_apple_indexes.hpp` - the Apple twin over its one device, so one shard.
- `nvme_indexes.hpp` - read-only views of the parts a RAM or VRAM sharded index saved, every query's walks over all parts on one executor thread, and key lookups as bulk scans of the key columns.
- `ram_reshard.hpp`, `vram_nvidia_reshard.hpp` and `vram_amd_reshard.hpp` - the freestanding reshard plans moving a sharded index's sources into its targets, several sources gathered into one target, one source cut into several, or one index recut in place under a shard capacity.
- `nvme_reshard.hpp` - the freestanding reshard, refine and build plans that write new parts from old ones in phases of tasks, within a workspace the caller lends, each target part holding at most a shard capacity of entries, and that round each part in RAM when it fits the workspace and streamed from the drives when it does not.

## Construction

Every residency builds through the same two verbs, in this order:

- `insert` wires each node against the graph as it stood, so a narrow `expansion_factor` ingests cheaply and leaves a graph whose picks a wider walk could improve on.
- `refine` runs rounds over the finished graph: a health check from a root, a heal linking each island it names into the root's component without a distance, a draw of hubs, a gather streaming every entry past them into the neighborhood of its nearest one, and a restitch of each neighborhood in one core's cache, closed by a second health check that decides the verdict.
  It claims no identity and moves no watermark, so a stopped round merely leaves the graph less improved.
  Call it until a round answers other than improved.
  `reshard` moves entries with their links from its sources into its targets, then runs the same rounds, so no verb walks to find what an entry links to.

`health` is the read-only check the heal starts from - the same sweep from the root, reporting the reachable population, its depths, islands and components, with no edit.
Use it to judge a graph without changing it.

Every sharded index and residency runs the same round and reports one `utilities::refine_report_t`, whose counts and verdicts mean the same everywhere.
The sharded index fans the rounds over one executor, each shard's on its own device in VRAM.
The RAM mono-index hands out plans instead - `plan_refine` and `plan_health` cut each phase into tasks any number of threads may run, with `advance` the one barrier between phases - since one index knows nothing about pools.
