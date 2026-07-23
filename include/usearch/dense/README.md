# Dense Indexes

Everything dense, one level deep: equidimensional vectors whose most common similarity measures - Inner-Product, Angular, Euclidean, Hamming, Jaccard - derive from a single inner product of two vectors, so pre-computed per-vector norms cut the arithmetic intensity of the composite metrics by 3x.
The flat glue header `usearch/dense_index.hpp` pulls the portable RAM residency; the VRAM and NVMe siblings stay opt-in through their own extensions.

## Residencies

- `ram_metric.hpp` + `ram_index.hpp` - the CPU baseline: compile-time-dispatched NumKong kernel kits chained per capability tier, and the concurrent, watermark-published in-memory index over them.
- `vram_metric.cuh` + `vram_index.cuh` - the CUDA sibling: per-generation device kits - `ampere` through `blackwell` ceilings - under one index whose geometry is frozen at the first `reserve`.
- `nvme_index.hpp` - the persisted sibling: a pure traversal state machine pumped by pluggable I/O engines, from the portable POSIX floor through `io_uring` rings to NVMe passthrough; its engine kits ride here until a second modality needs them.

## The Plural Layer

The plural follows the sibling idiom - `ram_indexes.hpp` reads as "many `ram_index` viewed as one":

- `ram_indexes.hpp` - typed multi-shard views with compute-domain routing and merged cross-shard search, one shard per memory domain.
- `vram_indexes.cuh` - multi-GPU placement over a shared address space, hosting the mergers & stitching machinery too.

Capability files split beside the residency pairs as they grow - `icelake.hpp`, `sapphireamx.hpp`, `hopper.cuh`, and their siblings.
