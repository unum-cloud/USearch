<h1 align="center">USearch</h1>
<h3 align="center">
Billion-Scale Multi-Modal <a href="https://ashvardanian.com/posts/abusing-vector-search/">Index</a><br/>
for CPUs, GPUs &amp; NVMe SSDs
</h3>
<br/>

<p align="center">
<a href="https://discord.gg/A6wxt6dS9j"><img height="25" src="https://github.com/unum-cloud/.github/raw/main/assets/discord.svg" alt="Discord"></a>
&nbsp;&nbsp;&nbsp;
<a href="https://www.linkedin.com/company/unum-cloud/"><img height="25" src="https://github.com/unum-cloud/.github/raw/main/assets/linkedin.svg" alt="LinkedIn"></a>
&nbsp;&nbsp;&nbsp;
<a href="https://twitter.com/unum_cloud"><img height="25" src="https://github.com/unum-cloud/.github/raw/main/assets/twitter.svg" alt="Twitter"></a>
&nbsp;&nbsp;&nbsp;
<a href="https://unum.cloud/post"><img height="25" src="https://github.com/unum-cloud/.github/raw/main/assets/blog.svg" alt="Blog"></a>
&nbsp;&nbsp;&nbsp;
<a href="https://github.com/unum-cloud/USearch"><img height="25" src="https://github.com/unum-cloud/.github/raw/main/assets/github.svg" alt="GitHub"></a>
</p>

<p align="center">
Dense Vectors • Sequences • Shapes • Geospatial Sets
<br/>
<a href="include/usearch/dense/README.md">C++</a> •
<a href="c/README.md">C</a> •
<a href="python/README.md">Python</a> •
<a href="rust/README.md">Rust</a> •
<a href="javascript/README.md">JavaScript</a> •
<a href="golang/README.md">Go</a> •
<a href="swift/README.md">Swift</a> •
<a href="java/README.md">Java, Kotlin &amp; Scala</a> •
<a href="csharp/README.md">C#</a> •
<a href="zig/README.md">Zig</a> •
<a href="mojo/README.md">Mojo</a> •
<a href="wolfram/README.md">Wolfram</a>
<br/>
Linux • macOS • Windows • FreeBSD • iOS • Android • WASI
<br/>
x86 • Arm • RISC-V • WebAssembly • CUDA • ROCm • Metal
</p>

<div align="center">
<a href="https://pepy.tech/project/usearch"> <img alt="PyPI" src="https://static.pepy.tech/personalized-badge/usearch?period=total&units=abbreviation&left_color=black&right_color=blue&left_text=Python%20PyPI%20installs"> </a>
<a href="https://www.npmjs.com/package/usearch"> <img alt="NPM" src="https://img.shields.io/npm/dy/usearch?label=JavaScript%20NPM%20installs"> </a>
<a href="https://crates.io/crates/usearch"> <img alt="Crate" src="https://img.shields.io/crates/d/usearch?label=Rust%20Crate%20installs"> </a>
<a href="https://www.nuget.org/packages/Cloud.Unum.USearch"> <img alt="NuGet" src="https://img.shields.io/nuget/dt/Cloud.Unum.USearch?label=CSharp%20NuGet%20installs"> </a>
<img alt="GitHub code size in bytes" src="https://img.shields.io/github/languages/code-size/unum-cloud/USearch?label=Repo%20size">
</div>

---

USearch is a widely deployed open-source search engine, powering Vector Search in [ClickHouse][clickhouse-code], [ScyllaDB](https://github.com/scylladb/vector-store), [DuckDB][duckdb-docs] & [YugabyteDB][yugabyte-code].

- ✅ __One Vector Index, Three Goals__ - Search, Sparse Matrix Multiplications, & Flash Attention folded into the graph walk.
- ✅ __Many Modalities__ - Dense Vectors, Sequences, Shapes, & Geospatial Sets, with experimental engines beyond Dense.
- ✅ __Three Residencies__ - RAM, VRAM, & NVMe, exchanging portable [SafeGraphs](docs/format.md) files.
- ✅ __128-bit Keys__ - Built around UUIDs for large-scale systems, with several vectors allowed under one key.
- ✅ __Atomic Visibility__ - RAM insertion batches publish complete rows & keys behind each shard's watermark.
- ✅ __Concurrent Mutability__ - RAM indexing, refinement, & queries can overlap.
- ✅ __14 Programming Languages__ - From IoT & mobile to GPU neoclouds, through one C ABI & a lower-level C++ core.

One of the most important features of USearch is using the hardware your machine already has, but most software leaves idle.
Even SIMD, the most vanilla of those features, often means multiplying dense `f32` vectors with AVX2.
For USearch, it meant developing one of the largest collections of mixed-precision numerical kernels & algebraic transforms in [NumKong][numkong]: `i8` dot-products with `i32` accumulation, packed single-bit arithmetic, & FP8 computations on CPUs that have no native FP8 instructions.
In slightly more detail:

- ✅ __SIMD__ - AVX2, WASM v128, LASX, AVX-512, NEON, RVV, Intel AMX, & Arm SME across NumKong's kernel collection.
- ✅ __Extended Atomics__ - Beyond the standard libraries of C++, Rust, or Zig, with [verification models](verification/README.md) for the index's concurrent protocols.
- ✅ __GPGPU__ - Generation-specific NVIDIA kits for Ampere, Ada, Hopper, Blackwell, & Blackwell RTX; AMD ROCm & Apple Metal backends are being integrated.
- ✅ __I/O Mechanisms__ - Blocking POSIX reads, Windows overlapped I/O, Linux AIO, `io_uring`, & SPDK in the storage engines.
- ✅ __Memory Pinning__ - Explicit NUMA placement for shards & workers; linking a different allocator doesn't place the graph near the threads walking it.

That barely begins to describe how many flavors of AVX-512 USearch distinguishes & dispatches under the hood.
The [architecture notes](include/usearch/dense/README.md) go deeper into the kits, residency layers, & core components you can use beyond the default public API.
The [verification notes](verification/README.md) cover the ordering & publication rules that a language's memory-safety guarantees alone cannot establish.

[clickhouse-code]: https://github.com/ClickHouse/ClickHouse/pull/53447
[clickhouse-docs]: https://clickhouse.com/docs/en/engines/table-engines/mergetree-family/annindexes
[duckdb-docs]: https://duckdb.org/2024/05/03/vector-similarity-search-vss.html
[yugabyte-code]: https://github.com/yugabyte/yugabyte-db/blob/366b9f5e3c4df3a1a17d553db41d6dc50146f488/src/yb/vector_index/usearch_wrapper.cc
[numkong]: https://github.com/ashvardanian/NumKong
[stringzilla]: https://github.com/ashvardanian/StringZilla
[forkunion]: https://github.com/ashvardanian/ForkUnion
[faiss]: https://github.com/facebookresearch/faiss
[cuvs]: https://github.com/rapidsai/cuvs

## Performance

Cross-engine comparisons need matched datasets, recall, hardware, & query batches.
[RetriEval](https://github.com/ashvardanian/RetriEval) compares USearch with [FAISS][faiss], [cuVS][cuvs], & full database products through one harness.

On this Apple M5 Pro, SIFT 10M builds in __7m 14s__ & searches at __75.15k QPS__ using 16 threads, `u8` storage, & Euclidean distance.
At connectivity 32 & expansion 256/96, __98.800%__ of queries recover the true nearest neighbor among ten results, & __96.703%__ of the true top-10 entries are recovered.
The [benchmark tables](bench/README.md#10m) record the configuration & checkpoint reloads.

## Architecture

An index dispatches on its Modality, Residency, stored scalar type, metric, & available hardware.
The build selects the kernel families it includes; runtime detection selects the newest eligible kit under the caller's capability ceiling.
The graph & its traversal are templated on those kits, so a walk carries the matching vector loads, dot-products, norms, atomics, & wait instructions together.

### Residencies

__RAM__ shards follow memory domains, __VRAM__ shards follow devices, & __NVMe__ views follow checkpoint parts.
Sharding partitions the indexed rows rather than replicating the entire graph.
[SafeGraphs](docs/format.md) carries the keys, vectors, norms, & links between those layers.

The useful unit of parallelism changes with residency.
CPU workers overlap independent walks; GPU warps cooperate on their candidates & vector fragments; NVMe workers keep many pending walks alive while reads complete.
Blackwell kits score candidate batches in `mma.sync` tiles fed by 256-bit loads, while storage walks explicitly yield the reads they need next.
`io_uring` submits those reads through kernel-managed rings; SPDK uses user-space storage drivers & polling.
The traversal's latency hiding depends on keeping useful work ready between memory accesses, device instructions, or I/O completions.

`search_if` makes sense in RAM & NVMe because a host predicate can admit results while excluded rows still route the walk.
An arbitrary host callback cannot run inside a device-resident GPU traversal.
`insert` grows RAM & VRAM indexes; NVMe views serve immutable parts, with separate build & refinement operations writing new parts.
See the [C++ guide](include/usearch/dense/README.md) for the layers & the [C guide](c/README.md) for the verbs served by each residency.

### Modalities

| Modality            | Inputs                                              | Similarities & Distances                                                  | Status                                   |
| :------------------ | :-------------------------------------------------- | :------------------------------------------------------------------------ | :--------------------------------------- |
| __Dense Vectors__   | Equal-width numeric rows or packed bit sets         | Inner-Product, Angular, Euclidean, Hamming, & Jaccard                     | RAM, VRAM, & NVMe engines                |
| __Sequences__       | Variable-length byte strings & UTF-8 text           | Levenshtein & overlap; declared global & local alignment                  | Experimental C++ engine & C interface    |
| __Shapes__          | 3D point clouds of any size                         | Chamfer, Hausdorff, Wasserstein-Procrustes, & Gromov-Wasserstein          | Experimental C++ engine & C interface    |
| __Geospatial Sets__ | WGS-84 points, point sets, & sampled paths          | Point proximity; declared Overlap, Shape, Coverage, & Route               | Experimental C++ engine & C interface    |

The experimental geospatial C interface currently uses a brute-force stand-in while its native engine & bindings develop.

## Dense Modality

Each shard supports up to roughly __4 billion entries__ through 32-bit internal node IDs.
Public keys remain 128-bit UUIDs; adjacency lists store those compact node IDs rather than repeating UUIDs on every edge.
Several shards extend the collection beyond one shard's address space.

The number of compiled storage types is controllable.
The default four are `bf16`, `u1`, `i8`, & `e4m3`; the full eleven add `f16`, `u8`, `e5m2`, `i4`, `u4`, `e3m2`, & `e2m3`.
Inputs can arrive as `f32` or `f64` & convert on ingestion.
Bit sets suit fingerprints; byte integers suit quantized embeddings; `e4m3` & `bf16` offer floating-point range at different memory costs.

Most dense measures reuse a dot-product & precomputed norms.
Squared Euclidean distance is $\lVert a\rVert^2 + \lVert b\rVert^2 - 2\langle a,b\rangle$; Angular distance derives from the normalized product; packed-bit intersections yield Hamming & Jaccard distances.
Geometry stays fixed once reserved, while the graph's population grows.
`refine` improves its links & `health` audits reachability, making disconnected regions visible in the same reports that measure recall & throughput.

### Search & MaxSim

`search` returns keys & distances from the walk's retained candidate pool.
`search_if` applies a result-admission predicate without severing the routing paths through rejected nodes.
Expansion controls the candidate pool, & connectivity controls the adjacency width; neither promises an exact nearest neighbor.

`maxsim` ranks multi-vector documents, with token rows sharing one document key.
For query tokens $q_i$ & document tokens $d_j$, it sums each query token's best similarity: $\sum_i \max_j \langle q_i,d_j\rangle$.
The implementation walks the query rows & reduces the observed token matches by key, sharing the graph machinery with ordinary retrieval.
This avoids flattening a document's token geometry into one pooled vector.

### Sparse Linear Algebra

Sparsity appears at several levels: fixed patterns in Tensor Cores, routed experts in MoE models, & structured operators in State Space Models.
USearch adds a data-dependent computation mask to an $M \times N \times K$ product: __Dense M, Sparse N, Dense K__.
The $M$ query rows choose a subset of the $N$ indexed rows through the graph, while each chosen product consumes all $K$ stored coordinates.

`multiply` exposes those products as a graph-selected portion of $A B^{\mathsf{T}}$.
It returns every pair scored by the traversal, including candidates the nearest-neighbor pool rejected, subject to an optional output bound.
Products are computed over the stored representation; unvisited entries remain unknown.
The returned offsets delimit query rows, keys identify the selected indexed rows, & scores contain their dot-products.

Exporting entries is one use of the walk.
A custom C++ fold can accumulate $\sum_{j\in S(q)} \langle q,b_j\rangle v_j$ directly, avoiding the product tape & its key exports.
With $v_j = w_j b_j$, a fixed selected set defines a sparse bilinear operator $B_S^{\mathsf{T}} W B_S$; applying a final vector forms $x^{\mathsf{T}} B_S^{\mathsf{T}} W B_S q$.
When the graph chooses $S(q)$ from the query, the resulting operation is adaptive rather than globally linear.

An index-to-index `join` could add sparsity along $M$ as well; that public operation is planned.
Sparsity along $K$ needs a compatible custom metric & representation through the core C++ components, beyond the current dense public API.

### Subquadratic Attention

`attend` folds values into an online Softmax as the graph scores keys.
Every scored key contributes, including keys outside the retained nearest-neighbor pool.
The fold maintains a running maximum, partition, & weighted value sum, so it materializes neither a full score matrix nor an intermediate top-k selection.
Several query heads share one traversal & one value lookup, with separate accumulators per head.

For the scored set $S(q)$, the output is $\frac{\sum_{j\in S(q)} \exp(s_j-m)v_j}{\sum_{j\in S(q)} \exp(s_j-m)}$.
It is normalized over that set & approximates full attention; the unvisited mass determines the remaining error.
The maximum, partition, & weighted sums can merge across shards without averaging already-normalized outputs.

With $T$ queries, $H$ heads, width $D$, & $W$ scored keys per query, the attention work scales with $T H W D$ instead of $T H N D$ for a dense scan.
That becomes subquadratic when $W$ grows sublinearly with the indexed population $N$.
Construction & maintenance have their own cost, & neither arbitrary data nor arbitrary attention distributions guarantee a small $W$ at a chosen accuracy.

## Sequence Modality

> [!WARNING]
> Experimental: the C interface declares alignment kinds the native engine does not serve yet.

The C++ RAM index works with variable-length byte sequences & UTF-8 text, using [StringZilla][stringzilla] for Levenshtein & overlap kernels, & the C interface exports it.
The declared C surface also covers Needleman-Wunsch global alignment, Smith-Waterman local alignment, & Gotoh affine gaps, which the engine refuses for now.
Sequence search is useful when edit operations & local matches carry more meaning than a dense embedding's geometry.

## Geospatial Modality

> [!WARNING]
> Experimental: the C++ point index is ahead of the broader Geospatial C interface.

The native index converts WGS-84 coordinates into Earth-Centered, Earth-Fixed positions stored as three centimeter-scale `i32` coordinates - __12 bytes per point__.
Multiple points can share one key to represent sampled paths or geographic sets.
Cheap chord distances guide candidate selection; ellipsoidal geodesic calculations refine those candidates where surface distance matters.
Chord ordering on an ellipsoid need not match geodesic ordering, so the candidate stage remains approximate.
The broader Overlap, Shape, Coverage, & Route calls are declared through the experimental C interface.

## Getting Started

Every binding is a thin layer over the same C ABI, so this matrix follows the C symbols each one calls.
The C++ core also exposes lower-level kits & custom folds.

| Operation                         |   C   | Python | Rust  |  JS   |  Go   | Swift | Java  |  C#   |  Zig  | Mojo  | Wolfram |
| :-------------------------------- | :---: | :----: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :-----: |
| Dense Init, Load, Insert & Search |   ✓   |   ✓    |   ✓   |   ✓   |   ✓   |   ✓   |   ✓   |   ✓   |   ✓   |   ✓   |    ✓    |
| Save, Metadata & Describe         |   ✓   |   ✓    |   ✓   |   ✓   |   ✓   |   ✓   |   ✓   |   ✓   |   ✓   |   ✓   |    ✓    |
| Refine & Health                   |   ✓   |   ✓    |   ✓   |   ✓   |   ✓   |   ✓   |   ✓   |   ✓   |   ✓   |   ✓   |    ✓    |
| Count & Get by Key                |   ✓   |   ✓    |   ✓   |   ✓   |   ✓   |   ✓   |   ✓   |   ✓   |   ✓   |   ✓   |    ✓    |
| Exact Scans                       |   ✓   |   ✓    |   ✓   |   ✓   |   ✓   |   ✓   |   ✓   |   ✓   |   ✓   |   ✓   |    ✓    |
| VRAM & NVMe Residencies           |   ✓   |   ✓    |   ✓   |   ✓   |   ✓   |   ✓   |   ✓   |   ✓   |   ✓   |   ✓   |    ✓    |
| Multiply & MaxSim                 |   ✓   |   ✓    |   ✓   |   ✓   |   ✓   |   ✓   |   ✓   |   ✓   |   ✓   |   ✓   |    ✓    |
| Folded Flash Attention            |   ✓   |   ✓    |   ✓   |   ✓   |   ✓   |   ✓   |   ✓   |   ✓   |   ✓   |   ✓   |    ✓    |
| Batched Iteration with Cursors    |   ✓   |   ✓    |   ✓   |   ✓   |   ✓   |   ✓   |   ✓   |   ✓   |   ✓   |   ✓   |    -    |
| Filtered Search with Predicates   |   ✓   |   ✓    |   ✓   |   -   |   ✓   |   ✓   |   -   |   ✓   |   ✓   |   ✓   |    -    |
| Deferred Calls                    |   ✓   |   ✓    |   ✓   |   ✓   |   ✓   |   ✓   |   ✓   |   ✓   |   ✓   |   ✓   |    -    |
| Merge & Streams ¹                 |   ✓   |   ✓    |   ✓   |   ✓   |   ✓   |   ✓   |   ✓   |   ✓   |   ✓   |   ✓   |    -    |
| Sequence, Shapes & Geospatial ²   |   ✓   |   -    |   -   |   -   |   -   |   -   |   -   |   -   |   -   |   -   |    -    |

✓ indicates a binding surface; engine availability still depends on residency, as detailed in the [C guide](c/README.md).
¹ Absorption runs in the C engine; streams expose the declared API, but their C engine verbs currently report unsupported.
² The experimental sequence, shape & geospatial C calls run the engines, on RAM & NVIDIA GPUs.
Kotlin & Scala consume the Java surface directly & match its column.

Most calls name an Index, an Executor for scheduling, & an Arena for outputs whose size grows with the input.
Typed buffers cross the ABI directly; arenas own the returned buffers & can be reused once their results are consumed.
The language guides above cover installation & complete examples.
See [contributing](CONTRIBUTING.md) for building from source, & [benchmarks](bench/README.md) for measuring it.

## Integrations

USearch is used in databases & frameworks:

- [x] ClickHouse: [C++](https://github.com/ClickHouse/ClickHouse/pull/53447), [docs][clickhouse-docs].
- [x] DuckDB: [post][duckdb-docs].
- [x] YugabyteDB: [C++][yugabyte-code].
- [x] ScyllaDB: [Rust](https://github.com/scylladb/vector-store), [presentation](https://www.slideshare.net/slideshow/vector-search-with-scylladb-by-szymon-wasik/276571548).
- [x] TiDB & TiFlash: [C++](https://github.com/pingcap/tiflash), [announcement](https://www.pingcap.com/article/introduce-vector-search-indexes-in-tidb/).
- [x] MemGraph: [C++](https://github.com/memgraph/memgraph/blob/784dd8520f65050d033aea8b29446e84e487d091/src/storage/v2/indices/vector_index.cpp), [announcement](https://memgraph.com/blog/simplify-data-retrieval-memgraph-vector-search).
- [x] Google: [UniSim](https://github.com/google/unisim), [RetSim](https://arxiv.org/abs/2311.17264) paper.
- [x] LangChain: [Python](https://github.com/langchain-ai/langchain/releases/tag/v0.0.257) & [JavaScript](https://github.com/langchain-ai/langchainjs/releases/tag/0.0.125).
- [x] Microsoft Semantic Kernel: [Python](https://github.com/microsoft/semantic-kernel/releases/tag/python-0.3.9.dev) & C#.
- [x] GPTCache: [Python](https://github.com/zilliztech/GPTCache/releases/tag/0.1.29).
- [x] Sentence-Transformers: Python [docs](https://www.sbert.net/docs/package_reference/quantization.html#sentence_transformers.quantization.semantic_search_usearch).
- [x] Pathway: [Rust](https://github.com/pathwaycom/pathway).
- [x] Vald: [GoLang](https://github.com/vdaas/vald).
- [x] MatrixOne: [GoLang](https://github.com/matrixorigin/matrixone).

USearch builds on sibling libraries: [NumKong][numkong] supplies every dense kernel, [StringZilla][stringzilla] the sequence kernels, & [ForkUnion][forkunion] the thread pools behind RAM & NVMe executors.
Lending an application's own ForkUnion pool through `usearch_executor_on_pool` is in progress.
[UStore](https://github.com/unum-cloud/UStore) pairing documents with their vectors, so both become visible in one transaction, is planned.

Related applications include [SwiftSemanticSearch](https://github.com/ashvardanian/SwiftSemanticSearch) for multimodal retrieval & [USearch Molecules](https://github.com/unum-science/USearchMolecules), indexing 7 billion molecules through 28 billion fingerprints.

## Citations

```bibtex
@software{Vardanian_USearch,
doi = {10.5281/zenodo.7949415},
author = {Vardanian, Ash},
title = {{USearch: Billion-Scale Multi-Modal Search, Clustering, & Sparse Algebra Engine for CPUs, GPUs, & NVMe Devices}},
url = {https://github.com/unum-cloud/USearch},
year = {2026},
}
```
