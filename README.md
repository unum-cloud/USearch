<h1 align="center">USearch</h1>
<h3 align="center">
One Proximity Graph for Search, Sparse Products &amp; Attention<br/>
across RAM, VRAM &amp; NVMe
</h3>
<br/>

<p align="center">
<a href="https://discord.gg/4mxGrenbNt"><img height="25" src="https://github.com/unum-cloud/.github/raw/main/assets/discord.svg" alt="Discord"></a>
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
<a href="https://github.com/unum-cloud/USearch/blob/main/include/usearch/dense/README.md">C++</a> •
<a href="https://github.com/unum-cloud/USearch/blob/main/c/README.md">C</a> •
<a href="https://github.com/unum-cloud/USearch/blob/main/python/README.md">Python</a> •
<a href="https://github.com/unum-cloud/USearch/blob/main/rust/README.md">Rust</a> •
<a href="https://github.com/unum-cloud/USearch/blob/main/javascript/README.md">JavaScript</a> •
<a href="https://github.com/unum-cloud/USearch/blob/main/golang/README.md">Go</a> •
<a href="https://github.com/unum-cloud/USearch/blob/main/swift/README.md">Swift</a> •
<a href="https://github.com/unum-cloud/USearch/blob/main/java/README.md">Java, Kotlin &amp; Scala</a> •
<a href="https://github.com/unum-cloud/USearch/blob/main/csharp/README.md">C#</a> •
<a href="https://github.com/unum-cloud/USearch/blob/main/zig/README.md">Zig</a> •
<a href="https://github.com/unum-cloud/USearch/blob/main/mojo/README.md">Mojo</a> •
<a href="https://github.com/unum-cloud/USearch/blob/main/wolfram/README.md">Wolfram</a>
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

![USearch banner](https://github.com/ashvardanian/ashvardanian/blob/master/repositories/USearch-v3.png?raw=true)

USearch is an open-source proximity-graph index for nearest-neighbor search over dense vectors, sequences, shapes & geospatial sets.
The same graph serves as a sparse operand, folding sparse matrix products, MaxSim scoring & softmax attention into its traversal, and one checkpoint opens in RAM, VRAM or on NVMe without being rewritten.
Its core is a header-only C++ 20 library with no external dependencies, a commitment to lightweight systems.
[ClickHouse][clickhouse-code], [DuckDB][duckdb-docs], [ScyllaDB](https://github.com/scylladb/vector-store) & [YugabyteDB][yugabyte-code] embed it for vector search.

- ✅ __One Vector Index, Three Goals__ - Search, Sparse Matrix Multiplications, & Flash Attention folded into the graph walk.
- ✅ __Many Modalities__ - Dense Vectors, Sequences, Shapes, & Geospatial Sets, with experimental engines beyond Dense.
- ✅ __Three Residencies__ - RAM, VRAM, & NVMe, exchanging portable [SafeGraphs](https://github.com/unum-cloud/USearch/blob/main/docs/format.md) files.
- ✅ __128-bit Keys__ - Built around UUIDs for large-scale systems, with several vectors allowed under one key.
- ✅ __Atomic Visibility__ - RAM insertion batches publish complete entries & keys behind each shard's watermark.
- ✅ __Concurrent Mutability__ - RAM queries overlap indexing & refinement, which take turns with each other.
- ✅ __14 Programming Languages__ - From IoT & mobile to GPU neoclouds, through one C ABI & a lower-level C++ core.

One of the most important features of USearch is using the hardware your machine already has, but most software leaves idle.
Even SIMD, the most vanilla of those features, often means multiplying dense `f32` vectors with AVX2.
For USearch, it meant developing one of the largest collections of mixed-precision numerical kernels & algebraic transforms in [NumKong][numkong]: `i8` dot-products with `i32` accumulation, packed single-bit arithmetic, & FP8 computations on CPUs that have no native FP8 instructions.
In slightly more detail:

- ✅ __SIMD__ - AVX2, WASM v128, LASX, AVX-512, NEON, RVV, Intel AMX, & Arm SME across NumKong's kernel collection.
- ✅ __Extended Atomics__ - Beyond the standard libraries of C++, Rust, or Zig, with [verification models](https://github.com/unum-cloud/USearch/blob/main/verification/README.md) for the index's concurrent protocols.
- ✅ __GPGPU__ - Generation-specific kits for NVIDIA Ampere, Ada, Hopper, Blackwell, & Blackwell RTX, AMD CDNA 3-5 through ROCm, & Apple GPUs through Metal, with GPUDirect Storage between NVMe drives and GPU memory.
- ✅ __I/O Mechanisms__ - Blocking POSIX reads, Windows overlapped I/O, Linux AIO, `io_uring`, & SPDK in the storage engines.
- ✅ __Memory Pinning__ - Explicit NUMA placement for shards & workers; linking a different allocator doesn't place the graph near the threads walking it.

That barely begins to describe how many flavors of AVX-512 USearch distinguishes & dispatches under the hood.
The [architecture notes](https://github.com/unum-cloud/USearch/blob/main/include/usearch/dense/README.md) go deeper into the kits, residency layers, & core components you can use beyond the default public API.
The [verification notes](https://github.com/unum-cloud/USearch/blob/main/verification/README.md) cover the ordering & publication rules that a language's memory-safety guarantees alone cannot establish.

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

On one NVIDIA B300 SXM6, SIFT 100M builds at __431,520 vectors/s__ & searches at __604,490 QPS__, with `u8` storage & Euclidean distance.
On 192 AWS Graviton5 cores, it builds at __161,952 vectors/s__ & searches at __173,823 QPS__, while [FAISS][faiss] HNSW builds at __84,536__ & searches at __178,600__.
Their 1-recall@10, the share of queries whose true nearest neighbor is among ten results, is __0.9913__, __0.9925__ & __0.9917__ respectively.
The [benchmark tables](https://github.com/unum-cloud/USearch/blob/main/bench/README.md#100m) record each row's expansions.

## Quickstart

Every binding is a thin layer over the same C ABI, so this matrix follows the C symbols each one calls.
The C++ core also exposes lower-level kits & custom folds.

| Operation                         |  C  | Python | Rust | JS  | Go  | Swift | Java | C#  | Zig | Mojo | Wolfram |
| :-------------------------------- | :-: | :----: | :--: | :-: | :-: | :---: | :--: | :-: | :-: | :--: | :-----: |
| Dense Init, Load, Insert & Search |  ✓  |   ✓    |  ✓   |  ✓  |  ✓  |   ✓   |  ✓   |  ✓  |  ✓  |  ✓   |    ✓    |
| Save, Metadata & Describe         |  ✓  |   ✓    |  ✓   |  ✓  |  ✓  |   ✓   |  ✓   |  ✓  |  ✓  |  ✓   |    ✓    |
| Refine & Health                   |  ✓  |   ✓    |  ✓   |  ✓  |  ✓  |   ✓   |  ✓   |  ✓  |  ✓  |  ✓   |    ✓    |
| Count & Get by Key                |  ✓  |   ✓    |  ✓   |  ✓  |  ✓  |   ✓   |  ✓   |  ✓  |  ✓  |  ✓   |    ✓    |
| Exact Scans                       |  ✓  |   ✓    |  ✓   |  ✓  |  ✓  |   ✓   |  ✓   |  ✓  |  ✓  |  ✓   |    ✓    |
| VRAM & NVMe Residencies           |  ✓  |   ✓    |  ✓   |  ✓  |  ✓  |   ✓   |  ✓   |  ✓  |  ✓  |  ✓   |    ✓    |
| Multiply & MaxSim                 |  ✓  |   ✓    |  ✓   |  ✓  |  ✓  |   ✓   |  ✓   |  ✓  |  ✓  |  ✓   |    ✓    |
| Folded Flash Attention            |  ✓  |   ✓    |  ✓   |  ✓  |  ✓  |   ✓   |  ✓   |  ✓  |  ✓  |  ✓   |    ✓    |
| Batched Iteration with Cursors    |  ✓  |   ✓    |  ✓   |  ✓  |  ✓  |   ✓   |  ✓   |  ✓  |  ✓  |  ✓   |    -    |
| Filtered Search with Predicates   |  ✓  |   ✓    |  ✓   |  -  |  ✓  |   ✓   |  -   |  ✓  |  ✓  |  ✓   |    -    |
| Deferred Calls                    |  ✓  |   ✓    |  ✓   |  ✓  |  ✓  |   ✓   |  ✓   |  ✓  |  ✓  |  ✓   |    -    |
| Merge & Streams ¹                 |  ✓  |   ✓    |  ✓   |  ✓  |  ✓  |   ✓   |  ✓   |  ✓  |  ✓  |  ✓   |    -    |
| Sequence, Shapes & Geospatial ²   |  ✓  |   -    |  -   |  -  |  -  |   -   |  -   |  -  |  -  |  -   |    -    |

✓ indicates a binding surface; engine availability still depends on residency, as detailed in the [C guide](https://github.com/unum-cloud/USearch/blob/main/c/README.md).
¹ Absorption runs in the C engine; streams expose the declared API, but their C engine verbs currently report unsupported.
² The experimental sequence, shape & geospatial C calls run the engines, on RAM & NVIDIA GPUs.
Kotlin & Scala consume the Java surface directly & match its column.

Most calls name an Index, an Executor for scheduling, & an Arena for outputs whose size grows with the input.
Typed buffers cross the ABI directly; arenas own the returned buffers & can be reused once their results are consumed.
The language guides above cover installation & complete examples.
See [contributing](https://github.com/unum-cloud/USearch/blob/main/CONTRIBUTING.md) for building from source, & [benchmarks](https://github.com/unum-cloud/USearch/blob/main/bench/README.md) for measuring it.

## Architecture

An index dispatches on its Modality, Residency, stored scalar type, metric, & available hardware.
The build selects the kernel families it includes; runtime detection picks the newest CPU, GPU, & NVMe classes the machine admits, unless the caller pins one.
A description reports the classes an index dispatches as & the `shards_count` it spans.
The graph & its traversal are templated on those kits, so a walk carries the matching vector loads, dot-products, norms, atomics, & wait instructions together.

### Residency

__RAM__ shards follow memory domains, __VRAM__ shards follow devices, & __NVMe__ views follow checkpoint parts.
Sharding partitions the indexed entries rather than replicating the entire graph.
[SafeGraphs](https://github.com/unum-cloud/USearch/blob/main/docs/format.md) carries the keys, vectors, norms, & links between those layers.

The useful unit of parallelism changes with residency.
CPU workers overlap independent walks; GPU warps cooperate on their candidates & vector fragments; NVMe workers keep many pending walks alive while reads complete.
Blackwell kits score candidate batches in `mma.sync` tiles fed by 256-bit loads, while storage walks explicitly yield the reads they need next.
`io_uring` submits those reads through kernel-managed rings; SPDK uses user-space storage drivers & polling.
The traversal's latency hiding depends on keeping useful work ready between memory accesses, device instructions, or I/O completions.

`search_if` makes sense in RAM & NVMe because a host predicate can admit results while excluded entries still route the walk.
An arbitrary host callback cannot run inside a device-resident GPU traversal.
`insert` grows RAM & VRAM indexes; NVMe views serve immutable parts, with separate build & refinement operations writing new parts.
See the [C++ guide](https://github.com/unum-cloud/USearch/blob/main/include/usearch/dense/README.md) for the layers & the [C guide](https://github.com/unum-cloud/USearch/blob/main/c/README.md) for the verbs served by each residency.

### Modalities

| Modality            | Inputs                                      | Similarities & Distances                                         | Status                                |
| :------------------ | :------------------------------------------ | :--------------------------------------------------------------- | :------------------------------------ |
| __Dense Vectors__   | Equal-width numeric rows or packed bit sets | Inner-Product, Angular, Euclidean, Hamming, & Jaccard            | RAM, VRAM, & NVMe engines             |
| __Sequences__       | Variable-length byte strings & UTF-8 text   | Levenshtein, overlap, & Needleman-Wunsch global alignment        | Experimental C++ engine & C interface |
| __Shapes__          | 3D point clouds of any size                 | Chamfer, Hausdorff, Wasserstein-Procrustes, & Gromov-Wasserstein | Experimental C++ engine & C interface |
| __Geospatial Sets__ | WGS-84 points, point sets, & sampled paths  | Point proximity; declared Overlap, Shape, Coverage, & Route      | Experimental C++ engine & C interface |

### Building Indexes

Indexes are built in two very different regimes, & USearch keeps them as separate operations.
Insertion is online & greedy: each entry commits to its neighbors once, against the graph as it stands.
Refinement & absorption are macro-level: they revisit entries already in the graph, judge the graph as a whole, & may spend more compute per entry in exchange for throughput.
A workflow that bulk-indexes first & bulk-serves later should run both; a workflow that streams entries into a live index may never refine at all.

#### Incremental Insertions

`insert` walks the graph from its entry points with a beam as wide as the expansion, picks a diverse subset of what the walk scored, & back-links each pick to the new entry.
It answers one question per entry - the approximate nearest neighbors of a new point in a fixed graph - so its cost is one latency-bound walk per entry, & entries become searchable as soon as their batch publishes.
The greediness is the price: the first entries chose their neighbors from a small graph, & no insertion revisits them.
A narrow insertion expansion ingests cheaply & leaves picks a wider walk would improve on.
An index grows as it inserts; `reserve` is only a hint of the capacity to come.

#### Macro-Level Refinements

`refine` re-selects neighbors for entries already in the graph, against the finished graph.
It solves a different problem from insertion: building an approximate nearest-neighbor graph over every entry at once, plus the global properties no single walk can see.
Its audit sweeps the graph from landmarks, finds islands no search reaches, & heals them.
Like compaction in a database, refinement is optional & never changes what the index contains, only how well it is connected, & it must never lower recall.
Because nothing waits on its answers, refinement can use bandwidth-oriented kernels a search cannot afford: on GPUs, a two-hop pass will score an entry's neighbors-of-neighbors as one Gram matrix on Tensor Cores, trading per-entry latency for many more distances per byte loaded.

#### Absorption & Compaction

`absorb` moves every entry of one or more donor indexes into a recipient, keeps their links, & re-selects only the entries across the new seams; the donors end empty.
Shards hold at most about 4 billion entries, so a collection past that size grows by adding shards, & `compact` absorbs small shards into larger ones while each stays under the bound.
Absorption also crosses residencies: GPU shards link to each other on their devices, then land in RAM as one graph.
NVMe parts are immutable, so their compaction writes new parts from old ones & leaves the sources untouched.

## Dense Modality

Each shard supports up to roughly __4 billion entries__ through 32-bit internal node IDs.
Public keys remain 128-bit UUIDs; adjacency lists store those compact node IDs rather than repeating UUIDs on every edge.
Several shards extend the collection beyond one shard's address space: `shard_capacity` bounds each, & a full generation of shards opens a new one on the same memory domains or devices.

The number of compiled storage types is controllable.
The default four are `bf16`, `u1`, `i8`, & `e4m3`; the full eleven add `f16`, `u8`, `e5m2`, `i4`, `u4`, `e3m2`, & `e2m3`.
Inputs can arrive as `f32` or `f64` & convert on ingestion.
Bit sets suit fingerprints; byte integers suit quantized embeddings; `e4m3` & `bf16` offer floating-point range at different memory costs.

Most dense measures reuse a dot-product & precomputed norms.
Squared Euclidean distance is $\lVert a\rVert^2 + \lVert b\rVert^2 - 2\langle a,b\rangle$; Angular distance derives from the normalized product; packed-bit intersections yield Hamming & Jaccard distances.
Geometry freezes when the index opens, while the graph's population grows.
`refine` improves its links & `health` audits reachability, making disconnected regions visible in the same reports that measure recall & throughput.

### Search & MaxSim

`search` returns keys & distances from the walk's retained candidates.
`search_if` applies a result-admission predicate without severing the routing paths through rejected nodes.
Expansion controls the candidates, & connectivity controls the adjacency width; neither promises an exact nearest neighbor.

`maxsim` ranks multi-vector documents, with token rows sharing one document key.
For query tokens $q_i$ & document tokens $d_j$, it sums each query token's best similarity: $\sum_i \max_j \langle q_i,d_j\rangle$.
The implementation walks the query rows to nominate documents, sharing the graph machinery with ordinary retrieval, then scores every token row of each nominated document exactly.
Each query lands a ragged run of its best `wanted` documents, or of every scored document when `wanted` is zero.
This avoids flattening a document's token geometry into one pooled vector.

### Sparse Linear Algebra

Sparsity appears at several levels: fixed patterns in Tensor Cores, routed experts in MoE models, & structured operators in State Space Models.
USearch adds a data-dependent computation mask to an $M \times N \times K$ product: __Dense M, Sparse N, Dense K__.
The $M$ query rows choose a subset of the $N$ indexed rows through the graph, while each chosen product consumes all $K$ stored coordinates.

`multiply` exposes those products as a graph-selected portion of $A B^{\mathsf{T}}$.
It returns every pair scored by the traversal, including candidates the nearest-neighbor walk rejected, subject to an optional output bound.
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
Every scored key contributes, including keys outside the retained nearest-neighbor candidates.
The fold maintains a running maximum, partition, & weighted value sum, so it materializes neither a full score matrix nor an intermediate top-k selection.
Several query heads share one traversal & one value lookup, with separate accumulators per head.

For the scored set $S(q)$, the output is $\frac{\sum_{j\in S(q)} \exp(s_j-m)v_j}{\sum_{j\in S(q)} \exp(s_j-m)}$.
It is normalized over that set & approximates full attention; the unvisited mass determines the remaining error.
The maximum, partition, & weighted sums can merge across shards without averaging already-normalized outputs.

With $T$ queries, $H$ heads, width $D$, & $W$ scored keys per query, the attention work scales with $T H W D$ instead of $T H N D$ for a dense scan.
That becomes subquadratic when $W$ grows sublinearly with the indexed population $N$.
Construction & maintenance have their own cost, & neither arbitrary data nor arbitrary attention distributions guarantee a small $W$ at a chosen accuracy.

## Sequence Modality

The C++ RAM index works with variable-length byte sequences & UTF-8 text, using [StringZilla][stringzilla] for Levenshtein & overlap kernels and its own for Needleman-Wunsch global alignment, & the C interface exports it.
Sequence search is useful when edit operations & local matches carry more meaning than a dense embedding's geometry.

## Geospatial Modality

> [!WARNING]
> Experimental: the C++ point index is ahead of the broader Geospatial C interface.

The native index converts WGS-84 coordinates into Earth-Centered, Earth-Fixed positions stored as three centimeter-scale `i32` coordinates - __12 bytes per point__.
Multiple points can share one key to represent sampled paths or geographic sets.
Cheap chord distances guide candidate selection; ellipsoidal geodesic calculations refine those candidates where surface distance matters.
Chord ordering on an ellipsoid need not match geodesic ordering, so the candidate stage remains approximate.
The broader Overlap, Shape, Coverage, & Route calls are declared through the experimental C interface.

## Integrations

USearch is used in databases, frameworks & applications:

- [x] ClickHouse: [C++](https://github.com/ClickHouse/ClickHouse/pull/53447), [docs][clickhouse-docs].
- [x] DuckDB: [post][duckdb-docs].
- [x] YugabyteDB: [C++][yugabyte-code], [architecture](https://www.yugabyte.com/blog/yugabytedb-vector-indexing-architecture/), [1B-vector benchmark](https://www.yugabyte.com/blog/benchmarking-1-billion-vectors-in-yugabytedb/).
- [x] ScyllaDB: [Rust](https://github.com/scylladb/vector-store), [presentation](https://www.slideshare.net/slideshow/vector-search-with-scylladb-by-szymon-wasik/276571548).
- [x] TiDB & TiFlash: [C++](https://github.com/pingcap/tiflash), [announcement](https://www.pingcap.com/article/introduce-vector-search-indexes-in-tidb/).
- [x] MemGraph: [C++](https://github.com/memgraph/memgraph/blob/784dd8520f65050d033aea8b29446e84e487d091/src/storage/v2/indices/vector_index.cpp), [announcement](https://memgraph.com/blog/simplify-data-retrieval-memgraph-vector-search).
- [x] Google: [Perch](https://github.com/google-research/perch-hoplite) bioacoustics DB, [UniSim](https://github.com/google/unisim), [RetSim](https://arxiv.org/abs/2311.17264) paper.
- [x] LangChain: [Python](https://github.com/langchain-ai/langchain/releases/tag/v0.0.257) & [JavaScript](https://github.com/langchain-ai/langchainjs/releases/tag/0.0.125).
- [x] Microsoft: [Semantic Kernel](https://github.com/microsoft/semantic-kernel/releases/tag/python-0.3.9.dev) & [vcpkg](https://github.com/microsoft/vcpkg/tree/master/ports/usearch).
- [x] GPTCache: [Python](https://github.com/zilliztech/GPTCache/releases/tag/0.1.29).
- [x] Hugging Face: Sentence-Transformers [docs](https://www.sbert.net/docs/package_reference/quantization.html#sentence_transformers.quantization.semantic_search_usearch), [quantization benchmarks](https://huggingface.co/blog/embedding-quantization).
- [x] MemMachine: [Python](https://github.com/MemMachine/MemMachine/blob/main/packages/server/src/memmachine_server/common/vector_store/vector_search_engine/usearch_engine.py).
- [x] Pathway: [Rust](https://github.com/pathwaycom/pathway).
- [x] Vald: [GoLang](https://github.com/vdaas/vald).
- [x] MatrixOne: [GoLang](https://github.com/matrixorigin/matrixone).
- [x] GreptimeDB: [Rust](https://github.com/GreptimeTeam/greptimedb/blob/main/src/index/src/vector.rs).
- [x] KurrentDB: [C#](https://github.com/kurrent-io/KurrentDB/blob/master/src/KurrentDB.Kontext/Indexing/USearch/USearchVectorStore.cs), [credits](https://github.com/kurrent-io/KurrentDB/blob/master/NOTICE.md).
- [x] Chroma: [Rust](https://github.com/chroma-core/chroma/blob/main/rust/index/src/usearch.rs), behind their RaBitQ-quantized SPANN index.
- [x] Arm MCP: [Python](https://github.com/arm/mcp/blob/main/arm_kb_search/loaders.py).
- [x] GPT4All: [C++](https://github.com/nomic-ai/gpt4all/blob/main/gpt4all-chat/src/database.cpp).
- [x] Ente: [Rust](https://github.com/ente-io/ente/blob/main/rust/crates/photos/src/ml/vector_db.rs), [post](https://ente.com/blog/vector-db/).
- [x] Tracy profiler: [C++](https://github.com/wolfpld/tracy/blob/master/profiler/src/profiler/TracyLlmEmbeddings.cpp), behind the embeddings index of its LLM assistant.
- [x] Copilot for Xcode by Intitni: [Swift](https://github.com/intitni/CopilotForXcode/blob/main/Tool/Sources/USearchIndex/UsearchIndex.swift).
- [x] LLM for Unity: [C#](https://github.com/undreamai/LLMUnity/tree/main/Runtime/RAG/usearch), [docs](https://undream.ai/LLMUnity/).
- [x] Music Assistant: [Python](https://github.com/music-assistant/server/blob/dev/music_assistant/providers/sonic_similarity/vectors.py), [docs](https://www.music-assistant.io/plugins/sonic-similarity/).

USearch builds on sibling libraries: [NumKong][numkong] supplies every dense kernel, [StringZilla][stringzilla] the sequence kernels, & [ForkUnion][forkunion] the thread pools behind RAM & NVMe executors.
Lending an application's own ForkUnion pool through `usearch_executor_on_pool` is in progress.
[UStore](https://github.com/unum-cloud/UStore) pairing documents with their vectors, so both become visible in one transaction, is planned.

Related applications include [SwiftSemanticSearch](https://github.com/ashvardanian/SwiftSemanticSearch) for multimodal retrieval & [USearch Molecules](https://github.com/unum-science/USearchMolecules), indexing 7 billion molecules through 28 billion fingerprints.

## Citation

```bibtex
@software{Vardanian_USearch,
doi = {10.5281/zenodo.7949415},
author = {Vardanian, Ash},
title = {{USearch: One Proximity Graph for Search, Sparse Products and Attention, across RAM, VRAM and NVMe}},
url = {https://github.com/unum-cloud/USearch},
}
```

## License

Licensed under the Apache License, Version 2.0.
See `LICENSE` for details.
