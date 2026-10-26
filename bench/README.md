# Benchmarking USearch

Cross-engine comparisons live in [RetriEval](https://github.com/ashvardanian/RetriEval).
It drives USearch, FAISS, cuVS and standalone databases through one harness over the BigANN binary layout, with shuffled parallel construction, incremental recall curves and machine-readable reports, and its README carries the dataset catalogue with download recipes and pre-computed ground truth.
Run RetriEval to answer _which engine_; run this directory to answer _which primitive_.

The suite here is internal.
It puts USearch's own building blocks against each other on the same dataset and the same machine: the single index against the sharded multi-index, RAM against VRAM, one stored dtype against another, one kernel kit against the next microarchitecture's, one wait instruction against another, one connectivity against another.
Every result is a throughput, a latency and a recall under one explicit configuration, so a change in the traversal, the kits or the executors shows up as a row that moved.

## The Suite

The index benchmarks of `bench/CMakeLists.txt` drive the library through its C interface, as users and bindings do, beside one sparse probe and the kernel microbenchmark under [Kernels](#kernels).
One binary serves every residency, CPU class and GPU class the library compiled, so `--residency`, `--cpu-class`, `--gpu-class` and `--nvme-class` choose what a run measures, and the report names the class dispatch picked.

-   `usearch_bench_dense_index_search` builds one dense index of any residency from a RetriEval dataset, searches it at every `--expansion-search` breadth and judges the hits against ground truth, against itself, or against an exact scan.
-   `usearch_bench_dense_index_algebra` builds one inner-product index the same way and measures the graph products `multiply` and the graph attention `attend` over it, attention judged against exact softmax attention.
-   `usearch_bench_sequence_index_search`, `usearch_bench_shapes_index_search` and `usearch_bench_geospatial_index_search` build one index of their modality, hold the file's last `--queries` entries out of it, and judge the hits against an exact scan of `--oracle-queries`.
-   `usearch_bench_dense_ram_index_search` instantiates one `dense::ram_index` itself, at `f32` or `bf16` storage and at the serial class or the newest one its host compiled, for what only a benchmark looks inside: plan stepping, saturated walks and islands, per-query latency percentiles, immutable mode, indexer heads and managed memory.

A bad value prints one line, like `--top-k="0" does not parse, expected a positive count` or `RETRIEVAL_TOP_K="0" does not parse, expected a positive count`, and exits with status 1.
Every run reports the build, the index's entries, bytes, shards and classes from `describe`, and per breadth the recall at one and at `--top-k`, the nDCG, the throughput, and the distances, visited members and restarts per query from the traversal stats.
A hit counts when its distance is within the exact k-th, so ties the oracle broke arbitrarily are not undercounted; against `--query-neighbors` a hit counts by key.
A ground truth is computed for the whole file, so `--first-vector` or `--max-base-vectors` reshapes the graph it describes and the harness drops it, printing a notice; `--self-search` judges connectivity instead and needs no ground truth at all.

### Building

Every number below comes from a Release build: the `host_release` preset for CPU runs, and `cuda_release` for GPU ones, narrowed to the host GPU's family code, like `-DUSEARCH_CUDA_ARCHITECTURES=100f` on a B200 or B300.
The library compiles the first `USEARCH_DENSE_DTYPES` stored dtypes, so these presets measure `bf16`, `u1`, `i8` and `e4m3` alone; add `-DUSEARCH_DENSE_DTYPES=11` to measure the rest.

```sh
cmake --preset host_release && cmake --build --preset host_release --target usearch_bench_dense_index_search usearch_bench_dense_index_algebra usearch_bench_dense_ram_index_search usearch_bench_dense_ram_metric
cmake --preset cuda_release -DUSEARCH_CUDA_ARCHITECTURES=100f && cmake --build build_cuda_release --target usearch_bench_dense_index_search usearch_bench_dense_index_algebra
```

Every binary lands at the top of its build tree.
Every benchmark is memory-hungry at 100M and above, so run it under a cgroup cap rather than trusting it to refuse.

```sh
systemd-run --user --scope -p MemoryMax=16G -p MemorySwapMax=0 -- build_host_release/usearch_bench_dense_index_search ...
```

### Running

Flags share RetriEval's names, and each one defaults to the environment variable RetriEval reads for it, `--top-k` to `RETRIEVAL_TOP_K`, so one environment drives both harnesses and the command line overrides it; `--help` prints the full list for each binary.

```sh
build_host_release/usearch_bench_dense_index_search \
    --base-vectors data/sift-10m/base.10M.u8bin \
    --query-vectors data/sift-10m/query.public.10K.u8bin \
    --query-neighbors data/sift-10m/groundtruth.public.10K.ibin \
    --stored-dtype u8 --metric euclidean --expansion-search 32,64,128
```

Every binary over the C interface reads the same placement flags.

-   `--residency` defaults to `ram`: `ram`, `vram` over the `USEARCH_DEVICES` ordinals, or `nvme`, which builds in RAM, saves and searches a view.
-   `--cpu-class` defaults to the CPU's own: the class to dispatch as, like `intel-icelake` or `serial`, admitting every rung whose features it holds.
-   `--gpu-class` defaults to the devices' own: the GPU class to dispatch as, like `nvidia-ampere`.
-   `--nvme-class` is derived by default: the NVMe class, `posix`, `linux-aio`, `linux-uring`, `linux-uring-passthrough`, `linux-spdk` or `windows-ioring`.
-   `--threads` defaults to the executor's own: the threads of a RAM or NVMe executor.
-   `--shards` defaults to one per memory domain or device: shards each index opens on, repeating the domains or devices named.
-   `--shard-capacity` defaults to unbounded: entries one shard holds before another shard generation opens.
-   `--checkpoint` defaults to none: a path to save the built index under and reopen it from in its residency; an NVMe run without one saves to the temporary directory.
-   `USEARCH_DEVICES` and `USEARCH_NVME_PATHS` - environment variables, not flags - name the USearch device ordinals a VRAM index places shards on and the namespaces an NVMe view serves from.

The dense binaries add their dataset and graph flags.

-   `--base-vectors` is required: base vectors in the RetriEval layout, one of `.fbin`, `.bf16bin`, `.f16bin`, `.i8bin`, `.u8bin`, `.u1x8bin`, `.b1bin`.
    A glob pattern like `base.part_*.b1bin` names a set RetriEval split into parts, read as one file in the glob's sorted order, so keys stay global rows past 2^32.
-   `--query-vectors` defaults to the first 1000 base vectors: query vectors of the same dimensions.
-   `--query-neighbors` defaults to none: ground truth, `.ibin`, of global base rows per query.
-   `--stored-dtype` defaults to the file's dtype, or `bf16` for an `.fbin`: the stored scalar, cast on ingest.
-   `--metric` defaults to `angular`: `inner`, `angular`, `euclidean`, `hamming` or `jaccard`; the last two for `u1`.
-   `--connectivity` and `--expansion-add` default to the library's derivations: the graph degree and the candidates kept while inserting.
-   `--expansion-search` defaults to `96`: the search breadths, like `32,64,128`, each judged over the one graph the run builds.
-   `--max-refines` defaults to `0`: refinement passes after ingestion, stopping once one changes nothing.
-   `--compact-capacity` defaults to none: entries per shard after compacting the built index in place, reported with the distances it computed.
-   `--top-k` defaults to `10`: matches judged per query.
-   `--first-vector` defaults to `0`: first dataset row viewed, keyed by its global row, to slice one file across runs.
    Over a glob the viewed rows must lie inside one part, so `--first-vector 4608000000 --max-base-vectors 512000000` builds the tenth 512M-row part keyed by its global rows.
-   `--max-base-vectors` defaults to the whole file: cap on ingested rows.
-   `--self-search` defaults to off: judge every stored vector as its own query; its variable takes `1`, `0`, `true` or `false`.
-   `--oracle-queries` defaults to `256`: queries the exact scan judges without `--query-neighbors`.
-   `--batch` defaults to `65536`: rows each insert or search call takes.

The sparse probe keeps the C++ harness's own flags, `--max-compute serial` choosing the serial class over the host's newest, and `--save`, `--load`, `--checkpoints-dir` and `--skip-vectors` naming its SafeGraphs monolith.

### Indexer Heads

The sparse probe can rank its searches by a lightning indexer, as DeepSeek V4 and GLM 5 pick sparse-attention keys: a key scores `Σₕ wₕ·ReLU(qₕ·k)` and sits at distance `1 − score`.
The indexer ranks floating-point keys under the dot kernel, and its walks navigate by the `--query-vectors` rows when given, or by `Σₕ wₕ·qₕ` otherwise.

-   `--heads` defaults to none: every query's heads as consecutive rows of any `.fbin`-family file, cast to the stored dtype; needs `--weights`.
-   `--weights` defaults to none: one row of signed head weights per query in any `.fbin`-family file, with the model's scale folded in.
-   `--oracle-queries` defaults to `16`: queries an exact `f64` oracle judges when no `--query-neighbors` gives their truth.

```sh
build_host_release/usearch_bench_dense_ram_index_search --base-vectors keys.bf16bin --heads queries.bf16bin \
    --weights weights.fbin --query-neighbors truth-global.ibin --metric dot
```

### Checkpoints

`--checkpoint` saves the built index as SafeGraphs and reopens it in the run's residency, so every search after it runs against what the C interface loads.
An index of one live shard saves one monolith; one of several saves a part per shard and commits a manifest at the path last, and a load opens one shard per part.
An NVMe run views the checkpoint it saved, and `--nvme-class` picks the class its rings submit through.
Judge GPU builds through a reload: a search issued right after a bulk build runs against a device still settling, and the checkpoint path is exactly what users reach.

```sh
build_host_release/usearch_bench_dense_index_search --base-vectors data/sift-10m/base.10M.u8bin \
    --query-vectors data/sift-10m/query.public.10K.u8bin --query-neighbors data/sift-10m/groundtruth.public.10K.ibin \
    --metric euclidean --residency nvme --checkpoint /mnt/nvme0/sift10m.safegraphs --nvme-class linux-uring
```

### Refinement

`--max-refines` inserts a refinement phase between ingestion and search: each pass re-selects every node's neighbors against the finished graph, and the phase stops early once a pass re-wires nothing.
This is what a cheap ingestion pays for - a narrow `--expansion-add` topped with one or two wide passes, rather than one expensive build.
Judge the two builds at matched distance computations rather than matched knobs; the report prints the count for every phase.

### Algebra

`usearch_bench_dense_index_algebra` builds the index from the same dataset, residency and class flags as the search binary, then runs `multiply` and `attend` instead of searching.
It pins the inner product, the only kind both verbs are defined over, and judges no ground truth.

```sh
build_host_release/usearch_bench_dense_index_algebra \
    --base-vectors data/wiki-1m/base.1M.fbin \
    --query-vectors data/wiki-1m/query.public.100K.fbin \
    --stored-dtype bf16 --oracle-queries 16
```

Every query row runs one `multiply`, reported as rows per second and scored pairs per row.
Every `--head-count` consecutive rows then form one token's heads, which share one `attend` walk, every key it scores entering each head's softmax, FlashAttention-style.
Attend reports tokens per second beside the keys scored per token, so runs compare at matched cost; it looks values up on the host, so it runs on RAM and NVMe.
The first `--oracle-queries` rows are then judged head by head: the relative L2 error of each attended row against exact `f64` attention over every viewed row.

-   `--values` defaults to a synthetic table: the `.bf16bin` value rows `attend` looks up by key, as wide as the keys.
-   `--head-count` defaults to `1`: the grouped-query heads per token; a trailing partial group is left out.
-   `--attend-scale` defaults to one over the root of the dimensions: the dot-to-logit factor.
-   `--oracle-queries` defaults to `16`: the queries judged against the exact oracle.
-   `--reference` defaults to none: `.f64bin` exact attention rows of the judged queries, skipping the host oracle's full pass.
-   `--pairs` defaults to none: a dump of every pair each judged query's `multiply` scored, per query a `u32` count and that many `u64` global rows beside their `f64` dots.

### Kernels

`usearch_bench_dense_ram_metric` times the distance kits alone, outside any index: USearch's 1-to-1 `distance` and 1-to-4 `distances`, the tile engine where the machine offers one, and NumKong's own kernel as the baseline, for every dtype, kernel and capability tier the build holds.
Candidates are drawn at random from a 256 MB pool, so they stream from RAM, and each row prints calls, distances, bytes and scalar operations per second.
Rows are named `kernel/scalar/caps/shape/dimensions`, like `dot/bf16/neon/1to4/768d`, with `numkong` as the baseline's tier.

```sh
build_host_release/usearch_bench_dense_ram_metric --filter 'dot/bf16/.*/1to[14]/768d' --time-limit 2s
```

-   `--seed` defaults to `42`: seed the row picks and the adjacency noise derive from, or `random`.
-   `--filter` defaults to none: rows to run, as an ECMAScript regex or else a substring.
-   `--warmup` defaults to `1s`: untimed run ahead of each row, like `200ms`.
-   `--time-limit` defaults to `10s`: timed run of each row, like `200ms`.

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
| `pubchem-maccs-*`, PubChem  |         `u1` |        166 | hamming   | 10M, 100M     | 10K queries            |
| `pubchem-ecfp4-*`, PubChem  |         `u1` |       2048 | hamming   | 10M, 100M     | 10K queries            |
| `gdb13-ecfp4-1b`, GDB-13    |         `u1` |       2048 | hamming   | 1B            | 10K queries            |
| `gdb13-maccs-1b`, GDB-13    |         `u1` |        166 | hamming   | 1B            | 10K queries            |
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

The PubChem and GDB-13 sets come from the USearchMolecules bucket through RetriEval's `retri-download-molecules`, with 10,000 queries sampled from the base and Hamming ground truth, and `--limit 10000000` cutting the 10M sets.

The other modalities keep their own entries, judged by the exact scan each binary runs.

| Dataset                             | Entries                                     | Metric                 | Scales        |
| :---------------------------------- | :------------------------------------------ | :--------------------- | :------------ |
| `leipzig-news-10m`, Leipzig Corpora | English news sentences                      | `levenshtein-utf8`     | 10M           |
| `fineweb-sentences-*`, FineWeb      | English web sentences, deduplicated         | `levenshtein-utf8`     | 100M, 1B      |
| `hg002-reads-*`, GIAB HG002         | 150-nt Illumina reads                       | `levenshtein`          | 10M, 100M, 1B |
| `pubchem-conformers-*`, PubChem 3D  | `f16` atoms, one conformer per molecule     | `chamfer`, `hausdorff` | 10M, 100M     |
| `gdb13-conformers-*`, GDB-13 3D     | `f16` atoms, one conformer per molecule     | `chamfer`, `hausdorff` | 10M, 100M, 1B |
| `real-conformers-*`, Enamine REAL   | `f16` atoms, one conformer per molecule     | `chamfer`, `hausdorff` | 10M, 100M     |
| `alphafold-ca-*`, AlphaFold DB      | `f16` Cα atoms, one backbone per protein    | `chamfer`, `hausdorff` | 10M, 100M     |
| `panoramax-*`, Panoramax            | street-level observations, 11 `f64` columns | near, sees, seen       | 10M, 100M     |

Sequence sets are text files with one entry per line, which `--base-texts` reads.
Leipzig publishes at most 1M sentences per corpus, so `leipzig-news-10m` joins ten yearly English news corpora.
Shape sets are one `.f16bin` of 3 columns with every cloud's points end to end and an `.ibin` of points per cloud, which `--base-points` and `--base-counts` read.
The conformer sets take conformer 0 of every embedded molecule, hydrogens included and centered, from the `.3D.parquet` shards under `pubchem/`, `gdb13/` and `real/` in the USearchMolecules bucket.
The AlphaFold sets keep each predicted structure's Cα atoms, centered, from AlphaFold DB.
A `.f16bin` header counts points in `u32`, at most 4,294,967,295 per file, so the larger shape sets are sharded as `base.shard_*.f16bin` beside `counts.shard_*.ibin`, natural-sorted, as RetriEval reads sharded inputs.
Geospatial sets keep each observation in the field order of `usearch_geospatial_observation_t`, one `.f64bin` row of 11 values with NaN wherever the source recorded nothing, beside a `.u64bin` of each row's source picture ID.
Panoramax publishes every picture's metadata across its federated instances as one weekly GeoParquet file, `https://api.panoramax.xyz/data/geoparquet/panoramax.parquet`, so its sets are a filter and a column mapping away.

## Results

Every table below holds one row per machine and core class, so a new machine is a new row and nothing else moves.
Every row is connectivity 32 and top-10 over the whole file ingested in one run.
Expansion lists the insertion and search expansions: the smallest pair reaching 99% recall@10 or more, or the defaults of 256 and 96 on rows measured before that rule.
A row never inserts faster than it searches: when insertions outpace queries, the insertion expansion rises and the search expansion falls until the row searches at least as fast at 99% recall@10.

### 10M

SIFT, `u8`, euclidean, 10,000,000 rows of 128 dimensions.

| Machine                                            | Class       | Expansion |   Insertions |      Queries | Recall@10 | Date       |
| :------------------------------------------------- | :---------- | --------: | -----------: | -----------: | --------: | :--------- |
| Intel i9-13950HX, 8P+16E cores, 1 domain, 4K pages | `haswell`   |  256 / 96 |   12.8 k IPS |   46.9 k QPS |     98.5% | 2026-09-02 |
| NVIDIA RTX 4060 Laptop, 24 SMs, 8 GB               | `ada`       |  256 / 96 |  114.5 k IPS |  210.9 k QPS |     98.6% | 2026-09-02 |
| Apple M5 Pro, 6S+12P cores, 16 threads, 16K pages  | `neonsdot`  |  256 / 96 |  23.05 k IPS |  75.15 k QPS |   98.800% | 2026-10-11 |
| Apple M5 Pro, 20 GPU cores, 64 GiB shared memory   | `metal`     |  256 / 96 |  72.96 k IPS | 176.48 k QPS |   98.860% | 2026-10-11 |
| NVIDIA B300 MIG 1g.34gb slice, 18 SMs, 30.5 GB     | `blackwell` |  256 / 96 |  59.89 k IPS | 235.81 k QPS |   97.726% | 2026-10-14 |
| NVIDIA B200, 148 SMs, 178.35 GB                    | `blackwell` |  256 / 96 | 522.31 k IPS | 512.23 k QPS |   97.630% | 2026-10-15 |
| NVIDIA B300 SXM6, 148 SMs, 267.68 GB               | `blackwell` | 256 / 128 | 558.51 k IPS |   1.04 M QPS |   99.110% | 2026-10-21 |

PubChem MACCS and ECFP4 from USearchMolecules, `u1`, Hamming, the first 10,000,000 molecules, of 166 and 2048 bits.

| Machine | Class | Dataset | Expansion | Insertions | Queries | Recall@10 | Date |
| :------ | :---- | :------ | --------: | ---------: | ------: | --------: | :--- |
| -       | -     | -       |         - |          - |       - |         - | -    |

### 100M

SIFT, `u8`, euclidean, 100,000,000 rows of 128 dimensions, and SpaceV, `i8`, euclidean, 100,000,000 rows of 100 dimensions.

| Machine                                           | Class       | Dataset | Expansion |   Insertions |      Queries | Recall@10 | Date       |
| :------------------------------------------------ | :---------- | :------ | --------: | -----------: | -----------: | --------: | :--------- |
| Apple M5 Pro, 6S+12P cores, 16 threads, 16K pages | `neonsdot`  | SIFT    |  256 / 96 |  16.86 k IPS |   5.63 k QPS |   96.410% | 2026-10-11 |
| Apple M5 Pro, 20 GPU cores, 64 GiB shared memory  | `metal`     | SIFT    |  256 / 96 |  61.99 k IPS | 146.82 k QPS |   97.650% | 2026-10-11 |
| NVIDIA B300 MIG 1g.34gb slice, 18 SMs, 30.5 GB    | `blackwell` | SIFT    |  256 / 96 |  56.52 k IPS | 214.77 k QPS |   95.190% | 2026-10-14 |
| NVIDIA B200, 148 SMs, 178.35 GB                   | `blackwell` | SIFT    |  256 / 96 | 342.23 k IPS | 483.06 k QPS |   95.160% | 2026-10-15 |
| NVIDIA B300 SXM6, 148 SMs, 267.68 GB              | `blackwell` | SIFT    | 256 / 256 | 431.52 k IPS | 604.49 k QPS |   99.130% | 2026-10-21 |

PubChem MACCS and ECFP4 from USearchMolecules, `u1`, Hamming, all 115,627,267 molecules, of 166 and 2048 bits.

| Machine                              | Class       | Dataset | Expansion |   Insertions |      Queries | Recall@10 | Date       |
| :----------------------------------- | :---------- | :------ | --------: | -----------: | -----------: | --------: | :--------- |
| NVIDIA B300 SXM6, 148 SMs, 267.68 GB | `blackwell` | MACCS   |  256 / 64 | 559.14 k IPS |   1.46 M QPS |   99.070% | 2026-10-21 |
| NVIDIA B300 SXM6, 148 SMs, 267.68 GB | `blackwell` | ECFP4   | 512 / 512 | 146.81 k IPS | 285.23 k QPS |   99.060% | 2026-10-21 |

### 1B

SIFT, `u8`, euclidean, 1,000,000,000 rows of 128 dimensions, and Deep, `f32` stored as `bf16`, euclidean, 1,000,000,000 rows of 96 dimensions, the goal being one hour of build on a dual-socket CPU.

| Machine | Class | Dataset | Expansion | Insertions | Queries | Recall@10 | Date |
| :------ | :---- | :------ | --------: | ---------: | ------: | --------: | :--- |
| -       | -     | -       |         - |          - |       - |         - | -    |

GDB-13 MACCS and ECFP4 from USearchMolecules, `u1`, Hamming, all 977 million molecules, of 166 and 2048 bits.

| Machine | Class | Dataset | Expansion | Insertions | Queries | Recall@10 | Date |
| :------ | :---- | :------ | --------: | ---------: | ------: | --------: | :--- |
| -       | -     | -       |         - |          - |       - |         - | -    |

### Multiply

Wiki, `f32` stored as `bf16`, dot, 1,000,000 rows of 256 dimensions, every one of the 100,000 query rows running one `multiply`.

| Machine | Class | Rows | Pairs per row | Date |
| :------ | :---- | ---: | ------------: | :--- |
| -       | -     |    - |             - | -    |

### Attend

The same index, every query row running one `attend` as a single head, judged on 16 queries against exact `f64` attention.

| Machine | Class | Queries | Keys per query | Relative L2 error | Date |
| :------ | :---- | ------: | -------------: | ----------------: | :--- |
| -       | -     |       - |              - |                 - | -    |

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
build_host_release/usearch_bench_dense_ram_index_search --huge-pages 16384 ...
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
