# Benchmarking USearch

Cross-engine comparisons live in [RetriEval](https://github.com/ashvardanian/RetriEval).
It drives USearch, FAISS, cuVS and standalone databases through one harness over the BigANN binary layout, with shuffled parallel construction, incremental recall curves and machine-readable reports, and its README carries the dataset catalogue with download recipes and pre-computed ground truth.
Run RetriEval to answer _which engine_; run this directory to answer _which primitive_.

The suite here is internal.
It puts USearch's own building blocks against each other on the same dataset and the same machine: the single index against the sharded multi-index, RAM against VRAM, one stored dtype against another, one kernel kit against the next microarchitecture's, one wait instruction against another, one connectivity against another.
Every result is a throughput, a latency and a recall under one explicit configuration, so a change in the traversal, the kits or the executors shows up as a row that moved.

## The Suite

Three executables come out of `bench/CMakeLists.txt`, one per residency and cardinality.

| Executable                        | Index                                            | What it isolates                                                                                      |
| :-------------------------------- | :----------------------------------------------- | :---------------------------------------------------------------------------------------------------- |
| `usearch_bench_dense_ram_index`   | one `dense_ram_index` on one memory domain       | kernel kits, wait instructions, connectivity and expansion, per-thread staging, latency percentiles   |
| `usearch_bench_dense_ram_indexes` | one `dense_ram_indexes` over every memory domain | what the C interface drives: admission, rotation across shards, batched verbs, SafeGraphs checkpoints |
| `usearch_bench_dense_vram_index`  | one `dense_vram_index` on one device             | batch-per-launch build and search, the kit generation ceiling, managed-memory planes                  |

Each binary prints a `Machine` block, an `Insertions` block and a `Queries` block to standard output and nothing else, so a run is its transcript.
The `Insertions` block reports the wall time, the throughput in IPS, the distances computed with the shares spent in refinement, the visited members and the restarts.
The `Queries` block reports the throughput in QPS, the latency, the distances and members per query, and the judgment.

### Building

Every number below comes from a Release build: the `release` preset for the two CPU binaries, and `release_cuda` for the CUDA one, narrowed to the host GPU's family code, like `-DUSEARCH_CUDA_ARCHITECTURES=100f` on a B200 or B300.
The CUDA bench compiles a generation's kits only when every listed architecture runs them, so the preset's own list measures the Ampere kits alone, whatever the GPU.
The `profile` preset keeps the same optimization and adds the symbols `perf` needs; `dev` is the `-O2` iteration tree and measures nothing.

```sh
cmake --preset release && cmake --build --preset release --target usearch_bench_dense_ram_index usearch_bench_dense_ram_indexes
cmake --preset release_cuda -DUSEARCH_CUDA_ARCHITECTURES=100f && cmake --build build_release_cuda --target usearch_bench_dense_vram_index
```

Every binary lands at the top of its build tree, so the CPU ones are in `build_release/` and the CUDA one in `build_release_cuda/`.
Every benchmark is memory-hungry at 100M and above, so run it under a cgroup cap rather than trusting it to refuse.

```sh
systemd-run --user --scope -p MemoryMax=16G -p MemorySwapMax=0 -- build_release/usearch_bench_dense_ram_indexes ...
```

### Running

The three binaries share one dataset triple and one tuning vocabulary; `--help` prints the full list for each.

```sh
build_release/usearch_bench_dense_ram_indexes \
    --vectors data/sift-10m/base.10M.u8bin \
    --queries data/sift-10m/query.public.10K.u8bin \
    --neighbors data/sift-10m/groundtruth.public.10K.ibin \
    --stored-dtype u8 --metric euclidean --connectivity 32 --expansion-add 256 --expansion-search 96
```

| Flag                 | Default                  | Meaning                                                                                                                                       |
| :------------------- | :----------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------- |
| `--vectors`          | required                 | Base vectors in the RetriEval layout: `.fbin`, `.bf16bin`, `.f16bin`, `.i8bin`, `.u8bin`, `.u1x8bin`                                          |
| `--queries`          | the first 1000 vectors   | Query vectors of the same dtype                                                                                                               |
| `--neighbors`        | none                     | Ground truth, `.ibin`; recall is skipped without it                                                                                           |
| `--stored-dtype`     | the file's dtype         | `f32`, `bf16`, `f16`, `e5m2`, `e4m3`, `e3m2`, `e2m3`, `i8`, `i4`, `u8`, `u4` or `u1`, cast on ingest; `f32` skips, as no CPU kit takes it yet |
| `--metric`           | `angular`                | `dot`, `angular`, `euclidean`, `hamming` or `jaccard`; the last two for `u1`                                                                  |
| `--connectivity`     | `32`                     | Graph degree: 16, 32 or 128                                                                                                                   |
| `--expansion-add`    | `256`                    | Candidate pool while inserting                                                                                                                |
| `--expansion-search` | `96`                     | Candidate pool while searching                                                                                                                |
| `--max-refines`      | `0`                      | Upper bound on refinement passes after ingestion                                                                                              |
| `--expansion-refine` | twice the connectivity   | Candidate pool while refining                                                                                                                 |
| `--top-k`            | `10`                     | Matches judged per query                                                                                                                      |
| `--max-vectors`      | the whole file           | Cap on ingested rows                                                                                                                          |
| `--self-recall`      | off                      | Judge every stored vector as its own query                                                                                                    |
| `--save`             | none                     | Checkpoint name to write after ingestion                                                                                                      |
| `--load`             | none                     | Checkpoint name to restore, then insert only the rows past `--skip-vectors`                                                                   |
| `--checkpoints-dir`  | `/tmp`                   | Directory for checkpoint blobs                                                                                                                |
| `--skip-vectors`     | `0`                      | Rows the loaded checkpoint already covers                                                                                                     |
| `--max-compute`      | the newest detected      | Kernel kit ceiling by microarchitecture, or `serial`; on the GPU by generation, or `ampere`                                                   |
| `--max-schedule`     | the newest offered       | Wait instruction ceiling, or `standard`; CPU only                                                                                             |
| `--threads`          | 7/8 of the logical cores | Executor threads; CPU only                                                                                                                    |
| `--memory-domain`    | the heap                 | NUMA domain to pin, or `all` on the multi-index; CPU only                                                                                     |
| `--batch-per-core`   | `4`                      | Staging rows per thread for a file cast on ingest; CPU only                                                                                   |
| `--first-vector`     | `0`                      | First dataset row viewed, to slice one file across runs; CPU only                                                                             |
| `--huge-pages`       | `0`                      | Huge pages provisioned before launch; the run fails rather than measuring base pages; CPU only                                                |
| `--device`           | `0`                      | CUDA device ordinal; GPU only                                                                                                                 |
| `--devices`          | none                     | Ordinals for the multi-device pipeline, like `0,1`; GPU only                                                                                  |
| `--merge`            | off                      | Stitch the device shards into one graph after ingest; GPU only                                                                                |
| `--expansion-merge`  | `--expansion-add`        | Candidate pool while merging; GPU only                                                                                                        |
| `--managed-memory`   | off                      | Managed-memory planes oversubscribing VRAM; GPU only                                                                                          |
| `--batch`            | `65536`                  | Rows and queries per launch; GPU only                                                                                                         |

A bad value prints one line, like `--top-k="0" does not parse, expected a positive count`, and exits with status 1.

Recall is judged against `--neighbors` as recall@1, recall@k, overlap@k and ndcg@k.
A ground truth is computed for the whole file, so `--max-vectors` reshapes the graph it describes and the harness drops it, printing a notice; `--self-recall` judges connectivity instead and needs no ground truth at all.
Judge GPU builds through a reload: a search issued right after a bulk build runs against a device still settling, and the checkpoint path is exactly what the C interface exposes.

### Refinement

`--max-refines` inserts a refinement phase between ingestion and search: each pass re-selects every node's neighbors against the finished graph, and the phase stops early once a pass re-wires nothing.
A heal follows, re-linking whatever no search can reach, and the report ends with the reachable population so a build that stranded nodes is visible rather than merely slow.

This is what a cheap ingestion pays for - a narrow `--expansion-add` topped with one or two wide passes, rather than one expensive build:

```sh
build_release_cuda/usearch_bench_dense_vram_index \
    --vectors data/sift-10m/base.10M.u8bin --queries data/sift-10m/query.public.10K.u8bin \
    --neighbors data/sift-10m/groundtruth.public.10K.ibin --stored-dtype u8 --metric euclidean \
    --connectivity 16 --expansion-add 32 --expansion-search 64 --max-refines 2 --expansion-refine 128
```

Judge the two builds at matched distance computations rather than matched knobs; the report prints the count for every phase.

### Checkpoints

A checkpoint is a name under `--checkpoints-dir`, stamped with the dtype, the metric and the connectivity, so any binary holding the same index shape restores it.
The GPU builds a base an order of magnitude faster than the CPU, so the fast way to a large CPU graph is a GPU base topped up on the CPU: build the first rows on the device, save, then reload on the CPU and insert the rest.

```sh
build_release_cuda/usearch_bench_dense_vram_index --vectors data/sift-100m/base.100M.u8bin --metric euclidean --max-vectors 90000000 --save sift90m
build_release/usearch_bench_dense_ram_index --vectors data/sift-100m/base.100M.u8bin --queries ... --neighbors ... --metric euclidean --load sift90m --skip-vectors 90000000
```

The second run reports `topping: 90000000 loaded rows`, inserts the remaining ten million, and judges the full graph against the 100M ground truth.
Today this crosses only between `usearch_bench_dense_vram_index` and `usearch_bench_dense_ram_index`, which share one checkpoint layout; `usearch_bench_dense_ram_indexes` writes SafeGraphs and restores a checkpoint only when it covers every row of the view.

### Kernels

`usearch_bench_dense_ram_metric` times the distance kits alone, outside any index: USearch's 1-to-1 `distance` and 1-to-4 `distances`, the tile engine where the machine offers one, and NumKong's own kernel as the baseline, for every dtype, kernel and capability tier the build holds.
Candidates are drawn at random from a 256 MB pool, so they stream from RAM, and each row prints calls, distances, bytes and scalar operations per second.
Rows are named `kernel/scalar/caps/shape/dimensions`, like `dot/bf16/neon/1to4/768d`, with `numkong` as the baseline's tier.

```sh
build_release/usearch_bench_dense_ram_metric --filter 'dot/bf16/.*/1to[14]/768d' --time-limit 2s
```

| Flag           | Default | Meaning                                                             |
| :------------- | :------ | :------------------------------------------------------------------ |
| `--seed`       | `42`    | Seed the row picks and the adjacency noise derive from, or `random` |
| `--filter`     | none    | Rows to run, as an ECMAScript regex or else a substring             |
| `--warmup`     | `1s`    | Untimed run ahead of each row, like `200ms`                         |
| `--time-limit` | `10s`   | Timed run of each row, like `200ms`                                 |

## Datasets

Datasets live outside the repository and are reached through `data/<name>` symlinks, lowercase with dashes and the scale suffix last, such as `data/sift-10m`.
The files follow the RetriEval layout: two little-endian `u32` counters, rows then columns, followed by the row-major payload, with the dtype carried by the extension, `.fbin` for `f32`, `.i8bin`, `.u8bin`, `.bf16bin`, and `.ibin` for `u32` neighbor ids.

| Dataset               | Stored dtype | Dimensions | Metric    | Scales        | Ground truth          |
| :-------------------- | -----------: | ---------: | :-------- | :------------ | :-------------------- |
| `wiki-1m`, UForm Wiki |        `f32` |        256 | dot       | 1M            | 100K queries          |
| `sift-*`, Meta BigANN |         `u8` |        128 | euclidean | 10M, 100M, 1B | 10K queries           |
| `spacev-*`, Microsoft |         `i8` |        100 | euclidean | 100M, 1B      | 30K queries           |
| `deep-*`, Yandex Deep |        `f32` |         96 | euclidean | 10M, 100M, 1B | 10K queries, 1B only  |
| `turing-*`, Microsoft |        `f32` |        100 | euclidean | 100M, 1B      | 100K queries          |
| `t2i-1b`, Yandex T2I  |        `f32` |        200 | angular   | 1B            | 100K queries, 1B only |

The BigANN base file has no pre-sliced subsets, so a subset is a range request followed by a header patch.

```sh
mkdir -p ~/datasets/sift-10m && ln -sfn ~/datasets/sift-10m data/sift-10m && cd data/sift-10m && \
    wget -nc https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/bigann/query.public.10K.u8bin && \
    wget -nc https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/GT_10M/bigann-10M -O groundtruth.public.10K.ibin && \
    wget --header="Range: bytes=0-1280000007" https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/bigann/base.1B.u8bin -O base.10M.u8bin && \
    printf '\x80\x96\x98\x00' | dd of=base.10M.u8bin bs=1 count=4 conv=notrunc status=none
```

The Yandex ground truth is published against the full 1B set only, so a 10M or 100M slice of Deep or T2I is judged by `--self-recall`.
Every other recipe is in RetriEval's README under the same names.

## Results

Every table below holds one row per machine and core class, so a new machine is a new row and nothing else moves.
Unless a row says otherwise the configuration is the default: connectivity 32, expansion 256 on insertion and 96 on search, top-10, the whole file ingested in one run.

### 10M

SIFT, `u8`, euclidean, 10,000,000 rows of 128 dimensions.

| Machine                                      | Class     |  Insertions |     Queries | Recall@10 | Date       |
| :------------------------------------------- | :-------- | ----------: | ----------: | --------: | :--------- |
| i9-13950HX, 8P+16E cores, 1 domain, 4K pages | `haswell` |  12.8 k IPS |  46.9 k QPS |     98.5% | 2026-09-02 |
| RTX 4060 Laptop, 24 SMs, 8 GB                | `ada`     | 114.5 k IPS | 210.9 k QPS |     98.6% | 2026-09-02 |

### 100M

SIFT, `u8`, euclidean, 100,000,000 rows of 128 dimensions, and SpaceV, `i8`, euclidean, 100,000,000 rows of 100 dimensions.
A 100M build needs about 50 GB resident at connectivity 32, so it belongs on a dual-socket machine with huge pages provisioned.

| Machine | Class | Dataset | Insertions | Queries | Recall@10 | Date |
| :------ | :---- | :------ | ---------: | ------: | --------: | :--- |

### 1B

SIFT, `u8`, euclidean, 1,000,000,000 rows of 128 dimensions, the goal being one hour of build on a dual-socket CPU.

| Machine | Class | Dataset | Insertions | Queries | Recall@10 | Date |
| :------ | :---- | :------ | ---------: | ------: | --------: | :--- |

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
build_release/usearch_bench_dense_ram_indexes --huge-pages 16384 ...
```

The pool splits per NUMA domain, so a single-domain run must fit its domain's share.

## Profiling

```sh
sudo -E perf stat -d build_profile/usearch_bench_dense_ram_index ...
sudo -E perf mem -d build_profile/usearch_bench_dense_ram_index ...
sudo -E perf record -F 1000 build_profile/usearch_bench_dense_ram_index ...
perf record -d -e arm_spe// -- build_profile/usearch_bench_dense_ram_index ...
sudo perf stat -e 'faults,dTLB-loads,dTLB-load-misses,cache-misses,cache-references' build_profile/usearch_bench_dense_ram_index ...
```

A dTLB miss rate near 3% and a cache-miss rate near 7% are typical of a healthy 1M run; stalls closer to 90% are the signal to provision huge pages and revisit allocation alignment.
