# Benchmarking USearch

Cross-engine comparisons live in [RetriEval](https://github.com/ashvardanian/RetriEval).
It drives USearch, FAISS, cuVS and standalone databases through one harness over the BigANN binary layout, with shuffled parallel construction, incremental recall curves and machine-readable reports, and its README carries the dataset catalogue with download recipes and pre-computed ground truth.
Run RetriEval to answer _which engine_; run this directory to answer _which primitive_.

The suite here is internal.
It puts USearch's own building blocks against each other on the same dataset and the same machine: the single index against the sharded multi-index, RAM against VRAM, one stored dtype against another, one kernel kit against the next microarchitecture's, one wait instruction against another, one connectivity against another.
Every result is a throughput, a latency and a recall under one explicit configuration, so a change in the traversal, the kits or the executors shows up as a row that moved.

## The Suite

The index benchmarks of `bench/CMakeLists.txt` come in two families that never share a binary, beside the kernel microbenchmark under [Kernels](#kernels).
Every index binary is named `usearch_bench_dense_<residency>_<arity>_<family>`: the residency is `ram`, `vram` or `nvme`, the arity a single `index` or sharded `indexes`, and the family `search` or `algebra`.
The `ram` and `vram` binaries of both families build the index the same way, from the same dataset, dtype, graph, refinement and checkpoint flags, and the `nvme` binaries view the parts they saved.

The search family searches the built index, by plain rows or by a lightning indexer's heads, and judges the hits.

- `usearch_bench_dense_ram_index_search` builds one `dense::ram_index` on one memory domain and isolates kernel kits, wait instructions, connectivity and expansion, per-thread staging and latency percentiles.
- `usearch_bench_dense_ram_indexes_search` builds one `dense::ram_indexes` over every memory domain and isolates what the C interface drives: admission, rotation across shards, batched verbs and SafeGraphs checkpoints.
- `usearch_bench_dense_vram_index_search` builds one `dense::vram_nvidia_index` on one device and isolates batch-per-launch build and search, the kit generation ceiling and managed-memory flat matrices.
- `usearch_bench_dense_vram_indexes_search` builds one `dense::vram_nvidia_indexes` with a shard per `USEARCH_DEVICES` ordinal and isolates what the C interface drives across GPUs: rotation across devices, scatter-gather search merged on the host, and SafeGraphs checkpoints.
- `usearch_bench_dense_nvme_indexes_search` views the checkpoint any RAM or VRAM binary saved as one `dense::nvme_indexes`, building nothing in RAM, and isolates what storage adds to the batched search: the I/O transport, the queue depth, and the reads and bytes each query costs.

The algebra family measures the graph products `multiply` and the graph attention `attend` over the built index instead of searching it.

- `usearch_bench_dense_ram_index_algebra` builds one `dense::ram_index` on one memory domain and isolates graph products and grouped-query graph attention over every key a walk scores, pair dumps and an exact softmax oracle.
- `usearch_bench_dense_ram_indexes_algebra` builds one `dense::ram_indexes` over every memory domain and isolates the same verbs batched through the executor, every token's walks crossing every shard.
- `usearch_bench_dense_vram_index_algebra` builds one `dense::vram_nvidia_index` on one device and isolates the same graph products and graph attention on the device, judged against the same oracle.
- `usearch_bench_dense_vram_indexes_algebra` builds one `dense::vram_nvidia_indexes` with a shard per `USEARCH_DEVICES` ordinal and isolates the same verbs across GPUs, the shards' attention merged on the first device.
- `usearch_bench_dense_nvme_indexes_algebra` views the same saved parts as one `dense::nvme_indexes` and isolates the same verbs answered from storage, with the reads and bytes each row costs.

The sequence and shapes binaries drive one RAM index through the C interface, as users do, holding the file's last `--queries` entries out of it, and judge the hits against an exact scan of `--oracle-queries`.
A hit counts when its distance is within the exact k-th, so edit distances, where many entries tie, are not undercounted.

- `usearch_bench_sequence_index_search` reads `--base-texts`, one entry per line, under `levenshtein`, `levenshtein-utf8` or `overlap` with `--window-widths`.
- `usearch_bench_shapes_index_search` reads `--base-points`, a `.fbin` or `.f16bin` of 3 columns stored in that scalar, cut by `--base-counts`, an `.ibin` of points per shape, under `chamfer`, `hausdorff`, `gromov_wasserstein` or `wasserstein_procrustes`.

Each dense search binary prints a `Machine` block, an `Insertions` block and a `Queries` block to standard output and nothing else, so a run is its transcript; the `nvme` one prints a `Storage` block in place of `Insertions`.
The `Insertions` block reports the wall time, the throughput in IPS, the distances computed with the shares spent in refinement, the visited members and the restarts.
The `Queries` block reports the throughput in QPS, the latency, the distances and members per query, and the judgment.

### Building

Every number below comes from a Release build: the `host_release` preset for the CPU binaries, and `cuda_release` for the CUDA ones, narrowed to the host GPU's family code, like `-DUSEARCH_CUDA_ARCHITECTURES=100f` on a B200 or B300.
The CUDA bench compiles a generation's kits only when every listed architecture runs them, so the preset's own list measures the Ampere kits alone, whatever the GPU.
The `host_profile` preset keeps the same optimization and adds the symbols `perf` needs; `cuda_relwithdebinfo` is the `-O2` iteration tree and measures nothing.
The per-dtype benchmark units follow `USEARCH_DENSE_DTYPES`, so these presets measure `bf16`, `u1`, `i8` and `e4m3` alone; add `-DUSEARCH_DENSE_DTYPES=11` to measure the rest.

```sh
cmake --preset host_release && cmake --build --preset host_release --target usearch_bench_dense_ram_index_search usearch_bench_dense_ram_indexes_search usearch_bench_dense_ram_index_algebra usearch_bench_dense_ram_indexes_algebra usearch_bench_dense_nvme_indexes_search usearch_bench_dense_nvme_indexes_algebra usearch_bench_dense_ram_metric
cmake --preset cuda_release -DUSEARCH_CUDA_ARCHITECTURES=100f && cmake --build build_cuda_release --target usearch_bench_dense_vram_index_search usearch_bench_dense_vram_indexes_search usearch_bench_dense_vram_index_algebra usearch_bench_dense_vram_indexes_algebra
```

Every binary lands at the top of its build tree, so the CPU ones are in `build_host_release/` and the CUDA ones in `build_cuda_release/`.
Every benchmark is memory-hungry at 100M and above, so run it under a cgroup cap rather than trusting it to refuse.

```sh
systemd-run --user --scope -p MemoryMax=16G -p MemorySwapMax=0 -- build_host_release/usearch_bench_dense_ram_indexes_search ...
```

### Running

The search binaries share one dataset triple and one tuning vocabulary, and the algebra binaries take the same build flags; `--help` prints the full list for each.
Flags share RetriEval's names, and each one defaults to the environment variable RetriEval reads for it, `--top-k` to `RETRIEVAL_TOP_K`, so one environment drives both harnesses and the command line overrides it.

```sh
build_host_release/usearch_bench_dense_ram_indexes_search \
    --base-vectors data/sift-10m/base.10M.u8bin \
    --query-vectors data/sift-10m/query.public.10K.u8bin \
    --query-neighbors data/sift-10m/groundtruth.public.10K.ibin \
    --stored-dtype u8 --metric euclidean --connectivity 32 --expansion-add 256 --expansion-search 96
```

- `--base-vectors` is required: base vectors in the RetriEval layout, one of `.fbin`, `.bf16bin`, `.f16bin`, `.i8bin`, `.u8bin`, `.u1x8bin`.
- `--query-vectors` defaults to the first 1000 base vectors: query vectors of the same dtype.
- `--query-neighbors` defaults to none: ground truth, `.ibin`; recall is skipped without it.
- `--stored-dtype` defaults to the file's dtype: `f32`, `bf16`, `f16`, `e5m2`, `e4m3`, `e3m2`, `e2m3`, `i8`, `i4`, `u8`, `u4` or `u1`, cast on ingest; `f32` skips, as no CPU kit takes it yet.
- `--metric` defaults to `angular`: `dot`, `angular`, `euclidean`, `hamming` or `jaccard`; the last two for `u1`.
- `--connectivity` defaults to `32`: the graph degree, 16, 32 or 128.
- `--expansion-add` defaults to `256`: candidate pool while inserting.
- `--expansion-search` defaults to `96`: candidate pool while searching.
- `--max-refines` defaults to `0`: upper bound on refinement passes after ingestion.
- `--expansion-refine` defaults to twice the connectivity: candidate pool while refining.
- `--top-k` defaults to `10`: matches judged per query.
- `--max-base-vectors` defaults to the whole file: cap on ingested rows.
- `--self-search` defaults to off: judge every stored vector as its own query; its variable takes `1`, `0`, `true` or `false`.
- `--save` defaults to none: checkpoint name to write after ingestion.
- `--load` defaults to none: checkpoint name to restore, then insert only the rows past `--skip-vectors`.
- `--checkpoints-dir` defaults to the temporary directory, which follows `TMPDIR`: directory for checkpoint blobs.
- `--skip-vectors` defaults to `0`: rows the loaded checkpoint already covers.
- `--inserts-per-core` defaults to `2048`: rows each insert call takes per CPU thread or GPU multiprocessor.
- `--queries-per-core` defaults to `2048`: queries each search call takes per CPU thread or GPU multiprocessor.
- `--max-compute` defaults to the newest detected: kernel kit ceiling by microarchitecture, or `serial`; on the GPU by generation, or `ampere`.
- `--max-schedule` defaults to the newest offered: wait instruction ceiling, or `standard`; CPU only.
- `--threads` defaults to 7/8 of the logical cores: executor threads; CPU only.
- `--memory-domain` defaults to the heap: NUMA domain to pin, or `all` on the multi-index; CPU only.
- `--first-vector` defaults to `0`: first dataset row viewed, to slice one file across runs; CPU only.
- `--huge-pages` defaults to `0`: huge pages provisioned before launch; the run fails rather than measuring base pages; CPU only.
- `USEARCH_DEVICES` - an environment variable, not a flag - names USearch device ordinals, positions in `usearch_gpu_devices`, like `0` or `0,1`; the single-device binaries take the first, the sharded ones place one shard per ordinal, and unset takes the first NVIDIA device; GPU only.
- `--merge` defaults to off: the inserted rows' second half builds as an index of its own, absorbed into the first half's before refinement and search, and the merge reports its wall time, distances and visits; `usearch_bench_dense_vram_indexes_search` only.
- `--expansion-merge` defaults to `--expansion-add`: candidate pool of the absorbing refine, whose rows walk across the appended part; only with `--merge`.
- `--managed-memory` defaults to off: managed-memory flat matrices oversubscribing VRAM; GPU only, refused by both `vram_indexes` binaries.

A bad value prints one line, like `--top-k="0" does not parse, expected a positive count` or `RETRIEVAL_TOP_K="0" does not parse, expected a positive count`, and exits with status 1.

Recall is judged against `--query-neighbors` as recall@1, recall@k, overlap@k and ndcg@k.
A ground truth is computed for the whole file, so `--max-base-vectors` reshapes the graph it describes and the harness drops it, printing a notice; `--self-search` judges connectivity instead and needs no ground truth at all.
Judge GPU builds through a reload: a search issued right after a bulk build runs against a device still settling, and the checkpoint path is exactly what the C interface exposes.

### Indexer Heads

Every search binary can rank its searches by a lightning indexer, as DeepSeek V4 and GLM 5 pick sparse-attention keys: a key scores `Σₕ wₕ·ReLU(qₕ·k)` and sits at distance `1 − score`.
The indexer ranks floating-point keys under the dot kernel, and its walks navigate by the `--query-vectors` rows when given, or by `Σₕ wₕ·qₕ` otherwise.

- `--heads` defaults to none: every query's heads as consecutive rows of any `.fbin`-family file, cast to the stored dtype; needs `--weights`.
- `--weights` defaults to none: one row of signed head weights per query in any `.fbin`-family file, with the model's scale folded in.
- `--oracle-queries` defaults to `16`: queries an exact `f64` oracle judges when no `--query-neighbors` gives their truth.

An indexer's selection is judged against the exact scan's top rows, saved once as `--query-neighbors`:

```sh
build_host_release/usearch_bench_dense_ram_index_search --base-vectors keys.bf16bin --heads queries.bf16bin \
    --weights weights.fbin --query-neighbors truth-global.ibin --metric dot
```

### Checkpoints

Every `ram` and `vram` binary, single or sharded, saves its index with `--save` as SafeGraphs, so any residency loads it with `--load` and the `nvme` binaries view it in place.
The name lands under `--checkpoints-dir`, stamped with the stored dtype, the metric and the connectivity, like `sift90m-u8-euclidean-c32.safegraphs`.
A single index writes that one monolith; a sharded one writes a monolith per shard and commits a `.safegraphs.json` manifest beside them last.
Keys are dataset rows, stored as `u32`, and widen on the way into a collection keyed `u64` or `uuid`, so the C interface opens a benchmark's checkpoint too.
The header names the metric, the value column's dtype names the stored scalar, and a `scale` entry records the factor a downcast rescaled the rows by.

The GPU builds a base an order of magnitude faster than the CPU, so the fast way to a large CPU graph is a GPU base topped up on the CPU: build the first rows on the device, save, then reload on the CPU and insert the rest.

```sh
build_cuda_release/usearch_bench_dense_vram_index_search --base-vectors data/sift-100m/base.100M.u8bin --metric euclidean --max-base-vectors 90000000 --save sift90m
build_host_release/usearch_bench_dense_ram_index_search --base-vectors data/sift-100m/base.100M.u8bin --query-vectors ... --query-neighbors ... --metric euclidean --load sift90m --skip-vectors 90000000
```

The second run reports `topping: 90000000 loaded rows`, inserts the remaining ten million, and judges the full graph against the 100M ground truth.
A single index also loads a sharded checkpoint, its parts appended in order, while a sharded index restores one only when it covers every row of the view.

Both `nvme_indexes` binaries build nothing: they view a checkpoint and take only the flags a view and its queries need.
The first part's header names the metric they dispatch on and the stored dtype, and its `scale` is what the queries are staged under.
The search judges its hits against `--query-neighbors` directly, and refuses a ground truth that names a row past the ones the parts hold.
Every query is answered from storage in one batched call, which arms as many rows per thread as the queue depth holds, and the verb times each query from its first armed walk to its final write.
The `Storage` block names the parts and the transport the engines opened, then each phase reports reads and kilobytes per query by purpose, the round trips, the IOPS and bandwidth it sustained, and the latency percentiles.

- `--view` is required: the monolith, or the manifest of the parts.
- `USEARCH_NVME_PATHS`, an environment variable like `USEARCH_DEVICES`, defaults to beside the manifest: one directory, file or device per part, comma-separated, to serve parts copied elsewhere.
- `--query-vectors` is required: query vectors in the RetriEval layout, cast to the parts' stored dtype.
- `--query-neighbors` is required by the search binary: ground truth, `.ibin`, over the rows the parts hold.
- `--transport` defaults to `auto`: `blocking`, `linux-aio` or `linux-uring`; `auto` takes io_uring, then AIO, then blocking reads, whichever the host admits first.
- `--queue-depth` defaults to `512`: reads each thread's engine keeps outstanding, which bounds the rows and the hops it overlaps.
- `--expansion-search`, `--top-k`, `--threads`, `--max-compute`, `--heads` and `--weights` read as in [Running](#running) and [Indexer Heads](#indexer-heads).

The algebra binary views only parts built for `dot`, looks the `--values` rows up by key, and runs its oracle over `--base-vectors`, the base file the keys index, which it needs only without a `--reference`.

```sh
cmake --preset host_release && cmake --build --preset host_release --target usearch_bench_dense_ram_indexes_search usearch_bench_dense_nvme_indexes_search usearch_bench_dense_nvme_indexes_algebra
build_host_release/usearch_bench_dense_ram_indexes_search --base-vectors data/sift-10m/base.10M.u8bin \
    --metric euclidean --save sift10m --checkpoints-dir /mnt/nvme0
build_host_release/usearch_bench_dense_nvme_indexes_search --view /mnt/nvme0/sift10m-u8-euclidean-c32.safegraphs.json \
    --query-vectors data/sift-10m/query.public.10K.u8bin \
    --query-neighbors data/sift-10m/groundtruth.public.10K.ibin --transport linux-uring --queue-depth 512
```

### Refinement

`--max-refines` inserts a refinement phase between ingestion and search: each pass re-selects every node's neighbors against the finished graph, and the phase stops early once a pass re-wires nothing.
A heal follows, re-linking whatever no search can reach, and the report ends with the reachable population so a build that stranded nodes is visible rather than merely slow.

This is what a cheap ingestion pays for - a narrow `--expansion-add` topped with one or two wide passes, rather than one expensive build:

```sh
build_cuda_release/usearch_bench_dense_vram_index_search \
    --base-vectors data/sift-10m/base.10M.u8bin --query-vectors data/sift-10m/query.public.10K.u8bin \
    --query-neighbors data/sift-10m/groundtruth.public.10K.ibin --stored-dtype u8 --metric euclidean \
    --connectivity 16 --expansion-add 32 --expansion-search 64 --max-refines 2 --expansion-refine 128
```

Judge the two builds at matched distance computations rather than matched knobs; the report prints the count for every phase.

### Algebra

The algebra binaries build the index exactly as the search binaries do, from the same dataset, dtype, executor, graph, refinement and checkpoint flags, and then run `multiply` and `attend` instead of the search phase.
They pin the dot kernel, the only one both verbs are defined over, so `--metric` takes only `dot` and no ground truth is judged.
They store only the dot kernel's dtypes, `bf16` through `u1`, so an `.fbin` file needs a `--stored-dtype`.

```sh
build_host_release/usearch_bench_dense_ram_index_algebra \
    --base-vectors data/wiki-1m/base.1M.fbin \
    --query-vectors data/wiki-1m/query.public.100K.fbin \
    --stored-dtype bf16 --oracle-queries 16
```

Every query row runs one `multiply`, reported as rows per second and scored pairs per row.
Every `--head-count` consecutive rows then form one query's heads, which share one `attend` walk per shard, every key it scores entering each head's softmax, FlashAttention-style.
Attend reports queries per second beside the keys scored per query, so runs compare at matched cost.
The first `--oracle-queries` rows are then judged head by head: the relative L2 error of each attended row against exact `f64` attention over every viewed row.

- `--values` defaults to a synthetic table: the `.bf16bin` value rows `attend` looks up by node or key, as wide as the keys.
- `--head-count` defaults to `1`: the grouped-query heads per query; a trailing partial group is left out.
- `--attend-scale` defaults to one over the root of the dimensions: the dot-to-logit factor.
- `--oracle-queries` defaults to `16`: the queries judged against the exact oracle.
- `--reference` defaults to none: `.f64bin` exact attention rows of the judged queries, skipping the host oracle's full pass.
- `--pairs` defaults to none: a dump of every pair each judged query's `multiply` scored; CPU only.

### Kernels

`usearch_bench_dense_ram_metric` times the distance kits alone, outside any index: USearch's 1-to-1 `distance` and 1-to-4 `distances`, the tile engine where the machine offers one, and NumKong's own kernel as the baseline, for every dtype, kernel and capability tier the build holds.
Candidates are drawn at random from a 256 MB pool, so they stream from RAM, and each row prints calls, distances, bytes and scalar operations per second.
Rows are named `kernel/scalar/caps/shape/dimensions`, like `dot/bf16/neon/1to4/768d`, with `numkong` as the baseline's tier.

```sh
build_host_release/usearch_bench_dense_ram_metric --filter 'dot/bf16/.*/1to[14]/768d' --time-limit 2s
```

- `--seed` defaults to `42`: seed the row picks and the adjacency noise derive from, or `random`.
- `--filter` defaults to none: rows to run, as an ECMAScript regex or else a substring.
- `--warmup` defaults to `1s`: untimed run ahead of each row, like `200ms`.
- `--time-limit` defaults to `10s`: timed run of each row, like `200ms`.

## Datasets

Datasets live outside the repository and are reached through `data/<name>` symlinks, lowercase with dashes and the scale suffix last, such as `data/sift-10m`.
The files follow the RetriEval layout: two little-endian `u32` counters, rows then columns, followed by the row-major payload, with the dtype carried by the extension, `.fbin` for `f32`, `.f16bin`, `.f64bin`, `.i8bin`, `.u8bin`, `.bf16bin`, `.b1bin` with columns counting bits, and `.ibin` for `u32` neighbor IDs.
Datasets and ground truth stay in this layout, while every index the benchmarks save is SafeGraphs, as [Checkpoints](#checkpoints) describes.

| Dataset                     | Stored dtype | Dimensions | Metric    | Scales        | Ground truth           |
| :-------------------------- | -----------: | ---------: | :-------- | :------------ | :--------------------- |
| `wiki-1m`, UForm Wiki       |        `f32` |        256 | dot       | 1M            | 100K queries           |
| `sift-*`, Meta BigANN       |         `u8` |        128 | euclidean | 10M, 100M, 1B | 10K queries            |
| `spacev-*`, Microsoft       |         `i8` |        100 | euclidean | 100M, 1B      | 30K queries            |
| `deep-*`, Yandex Deep       |        `f32` |         96 | euclidean | 10M, 100M, 1B | 10K queries, 1B only   |
| `turing-*`, Microsoft       |        `f32` |        100 | euclidean | 100M, 1B      | 100K queries           |
| `t2i-1b`, Yandex T2I        |        `f32` |        200 | angular   | 1B            | 100K queries, 1B only  |
| `pubchem-ecfp4-*`, PubChem  |         `u1` |       2048 | jaccard   | 10M, 100M     | 10K queries, 100M only |
| `pubchem-maccs-*`, PubChem  |         `u1` |        168 | jaccard   | 10M, 100M     | 10K queries, 100M only |
| `pubchem-usrcat-*`, PubChem |        `f16` |         60 | euclidean | 10M, 100M     | 10K queries, 100M only |

The BigANN base file has no pre-sliced subsets, so a subset is a range request followed by a header patch.

```sh
mkdir -p ~/datasets/sift-10m && ln -sfn ~/datasets/sift-10m data/sift-10m && cd data/sift-10m && \
    wget -nc https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/bigann/query.public.10K.u8bin && \
    wget -nc https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/GT_10M/bigann-10M -O groundtruth.public.10K.ibin && \
    wget --header="Range: bytes=0-1280000007" https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/bigann/base.1B.u8bin -O base.10M.u8bin && \
    printf '\x80\x96\x98\x00' | dd of=base.10M.u8bin bs=1 count=4 conv=notrunc status=none
```

The Yandex ground truth is published against the full 1B set only, so a 10M or 100M slice of Deep or T2I is judged by `--self-search`.
Every other recipe is in RetriEval's README under the same names.

The PubChem sets come from the USearchMolecules bucket, 99,964,587 base molecules with 10,000 held-out queries, Tanimoto ties judged by distance rather than by ID.

The other modalities keep their own entries, judged by the exact scan each binary runs.

| Dataset                             | Entries                                     | Metric                 | Scales        |
| :---------------------------------- | :------------------------------------------ | :--------------------- | :------------ |
| `leipzig-news-10m`, Leipzig Corpora | English news sentences                      | `levenshtein-utf8`     | 10M           |
| `fineweb-sentences-*`, FineWeb      | English web sentences, deduplicated         | `levenshtein-utf8`     | 100M, 1B      |
| `hg002-reads-*`, GIAB HG002         | 150-nt Illumina reads                       | `levenshtein`          | 10M, 100M, 1B |
| `pubchem-conformers-*`, PubChem 3D  | `f16` atoms, one conformer per molecule     | `chamfer`, `hausdorff` | 10M, 100M     |
| `gdb13-conformers-1b`, GDB-13 3D    | `f16` atoms, one conformer per molecule     | `chamfer`, `hausdorff` | 1B            |
| `panoramax-*`, Panoramax            | street-level observations, 11 `f64` columns | near, sees, seen       | 10M, 100M     |

Sequence sets are text files with one entry per line, which `--base-texts` reads.
Leipzig publishes at most 1M sentences per corpus, so `leipzig-news-10m` joins ten yearly English news corpora.
Shape sets are one `.f16bin` of 3 columns with every cloud's points end to end and an `.ibin` of points per cloud, which `--base-points` and `--base-counts` read.
Geospatial sets keep each observation in the field order of `usearch_geospatial_observation_t`, one `.f64bin` row of 11 values with NaN wherever the source recorded nothing, beside a `.u64bin` of each row's source picture ID.
Panoramax publishes every picture's metadata across its federated instances as one weekly GeoParquet file, `https://api.panoramax.xyz/data/geoparquet/panoramax.parquet`, so its sets are a filter and a column mapping away.

## Results

Every table below holds one row per machine and core class, so a new machine is a new row and nothing else moves.
Unless a row says otherwise the configuration is the default: connectivity 32, expansion 256 on insertion and 96 on search, top-10, the whole file ingested in one run.

### 10M

SIFT, `u8`, euclidean, 10,000,000 rows of 128 dimensions.

| Machine                                            | Class       |   Insertions |      Queries | Recall@10 | Date       |
| :------------------------------------------------- | :---------- | -----------: | -----------: | --------: | :--------- |
| Intel i9-13950HX, 8P+16E cores, 1 domain, 4K pages | `haswell`   |   12.8 k IPS |   46.9 k QPS |     98.5% | 2026-09-02 |
| NVIDIA RTX 4060 Laptop, 24 SMs, 8 GB               | `ada`       |  114.5 k IPS |  210.9 k QPS |     98.6% | 2026-09-02 |
| Apple M5 Pro, 6S+12P cores, 16 threads, 16K pages  | `neonsdot`  |  23.05 k IPS |  75.15 k QPS |   98.800% | 2026-10-11 |
| Apple M5 Pro, 20 GPU cores, 64 GiB shared memory   | `metal`     |  72.96 k IPS | 176.48 k QPS |   98.860% | 2026-10-11 |
| NVIDIA B300 MIG 1g.34gb slice, 18 SMs, 30.5 GB     | `blackwell` |  59.89 k IPS | 235.81 k QPS |   97.726% | 2026-10-14 |
| NVIDIA B200, 148 SMs, 178.35 GB                    | `blackwell` | 522.31 k IPS | 512.23 k QPS |   97.630% | 2026-10-15 |

### 100M

SIFT, `u8`, euclidean, 100,000,000 rows of 128 dimensions, and SpaceV, `i8`, euclidean, 100,000,000 rows of 100 dimensions.

| Machine                                           | Class       | Dataset |   Insertions |      Queries | Recall@10 | Date       |
| :------------------------------------------------ | :---------- | :------ | -----------: | -----------: | --------: | :--------- |
| Apple M5 Pro, 6S+12P cores, 16 threads, 16K pages | `neonsdot`  | SIFT    |  16.86 k IPS |   5.63 k QPS |   96.410% | 2026-10-11 |
| Apple M5 Pro, 20 GPU cores, 64 GiB shared memory  | `metal`     | SIFT    |  61.99 k IPS | 146.82 k QPS |   97.650% | 2026-10-11 |
| NVIDIA B300 MIG 1g.34gb slice, 18 SMs, 30.5 GB    | `blackwell` | SIFT    |  56.52 k IPS | 214.77 k QPS |   95.190% | 2026-10-14 |
| NVIDIA B200, 148 SMs, 178.35 GB                   | `blackwell` | SIFT    | 342.23 k IPS | 483.06 k QPS |   95.160% | 2026-10-15 |

### 1B

SIFT, `u8`, euclidean, 1,000,000,000 rows of 128 dimensions, the goal being one hour of build on a dual-socket CPU.

| Machine | Class | Dataset | Insertions | Queries | Recall@10 | Date |
| :------ | :---- | :------ | ---------: | ------: | --------: | :--- |
| - | - | - | - | - | - | - |

## Hyper-Parameters

Every HNSW implementation shares the same three knobs under different names, and the defaults differ enough to invalidate a comparison made at defaults.

| Library   | Connectivity | Expansion on add | Expansion on search |
| :-------- | -----------: | ---------------: | ------------------: |
| `hnswlib` |           16 |              200 |                  10 |
| `FAISS`   |           32 |               40 |                  16 |
| `USearch` |           32 |              256 |                  96 |

## Huge Pages

Graph traversal is bound by memory latency, and on a dual-socket Sapphire Rapids huge pages alone were worth 29% of build throughput and 20% of search throughput at connectivity 32.
Provision them before launch and declare them with `--huge-pages`, so a run that fell back to base pages fails instead of quietly measuring the wrong thing.

```sh
sudo sysctl -w vm.nr_hugepages=16384
build_host_release/usearch_bench_dense_ram_indexes_search --huge-pages 16384 ...
```

The pool splits per NUMA domain, so a single-domain run must fit its domain's share.

## Profiling

```sh
sudo -E perf stat -d build_host_profile/usearch_bench_dense_ram_index_search ...
sudo -E perf mem -d build_host_profile/usearch_bench_dense_ram_index_search ...
sudo -E perf record -F 1000 build_host_profile/usearch_bench_dense_ram_index_search ...
perf record -d -e arm_spe// -- build_host_profile/usearch_bench_dense_ram_index_search ...
sudo perf stat -e 'faults,dTLB-loads,dTLB-load-misses,cache-misses,cache-references' build_host_profile/usearch_bench_dense_ram_index_search ...
```

A dTLB miss rate near 3% and a cache-miss rate near 7% are typical of a healthy 1M run; stalls closer to 90% are the signal to provision huge pages and revisit allocation alignment.
