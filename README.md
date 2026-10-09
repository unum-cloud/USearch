<h1 align="center">USearch</h1>
<h3 align="center">
Billion-Scale Multi-Modal <a href="https://www.unum.cloud/blog/2023-11-07-scaling-vector-search-with-intel">Search</a>, Clustering, & Sparse Algebra<br/> Engine for <a href="https://github.com/ashvardanian/numkong">Vectors</a>, <a href="https://github.com/ashvardanian/stringzilla">Sequences</a> & Coordinates, across CPUs, GPUs & NVMe
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
Dense Vectors • Strings • Geo-Spatial • Meshes • User-Defined Metrics
<br/>
<a href="https://unum-cloud.github.io/USearch/cpp">C++</a> •
<a href="https://unum-cloud.github.io/USearch/c">C</a> •
<a href="https://unum-cloud.github.io/USearch/python">Python</a> •
<a href="https://unum-cloud.github.io/USearch/rust">Rust</a> •
<a href="https://unum-cloud.github.io/USearch/zig">Zig</a> •
<a href="https://unum-cloud.github.io/USearch/mojo">Mojo</a> •
<a href="https://unum-cloud.github.io/USearch/java">Java, Scala, & Kotlin</a> •
<a href="https://unum-cloud.github.io/USearch/javascript">JavaScript</a> •
<a href="https://unum-cloud.github.io/USearch/swift">Swift</a> •
<a href="https://unum-cloud.github.io/USearch/csharp">C#</a> •
<a href="https://unum-cloud.github.io/USearch/golang">GoLang</a> •
<a href="https://unum-cloud.github.io/USearch/wolfram">Wolfram</a>
<br/>
Linux • macOS • Windows • FreeBSD • iOS • Android • Browser
<br/>
x86 • Arm • WASM • RISC-V • LoongArch • Power • CUDA • ROCm • Metal
</p>

<div align="center">
<a href="https://pepy.tech/project/usearch"> <img alt="PyPI" src="https://static.pepy.tech/personalized-badge/usearch?period=total&units=abbreviation&left_color=black&right_color=blue&left_text=Python%20PyPI%20installs"> </a>
<a href="https://www.npmjs.com/package/usearch"> <img alt="NPM" src="https://img.shields.io/npm/dy/usearch?label=JavaScript%20NPM%20installs"> </a>
<a href="https://crates.io/crates/usearch"> <img alt="Crate" src="https://img.shields.io/crates/d/usearch?label=Rust%20Crate%20installs"> </a>
<a href="https://www.nuget.org/packages/Cloud.Unum.USearch"> <img alt="NuGet" src="https://img.shields.io/nuget/dt/Cloud.Unum.USearch?label=CSharp%20NuGet%20installs"> </a>
<!-- Maven Central publishing is deprecated for now; fat-JAR download is the supported path. -->
<img alt="GitHub code size in bytes" src="https://img.shields.io/github/languages/code-size/unum-cloud/USearch?label=Repo%20size">
</div>

---

- ✅ __[10-100× faster][faster-than-faiss]__ indexing & search than [FAISS][faiss].
- ✅ Powers SpGEMM and Sparse Attention for billion-scale transformer LLM context lengths.
- ✅ Simple and extensible [C++20][usearch-header] __library__, with one C ABI under all twelve bindings.
- ✅ [Trusted](#integrations--users) by most dabases like [ClickHouse][clickhouse-docs] & [DuckDB][duckdb-docs] & AI Frontier labs.
- ✅ [SIMD][simd]- and [GPU][gpgpu]-optimized and [user-defined metrics](#user-defined-functions) with JIT compilation.
- ✅ Hardware-agnostic `bf16`, `e5m2`, & `i8` - [half-precision & quarter-precision support](#memory-efficiency-downcasting-and-quantization).
- ✅ [Stored in RAM, on a GPU, or on NVMe](#serialization--serving-an-index-from-disk) - one checkpoint, three residencies.
- ✅ Heterogeneous lookups and 128-bit keys, with several entries free to share one.
- ✅ Binary Tanimoto and Sorensen coefficients for [Genomics and Chemistry applications](#usearch--rdkit--molecular-search).
- ✅ Quantization from `f64` down to packed nibbles and single bits, chosen per row rather than per build.
- ✅ Fine-grained parallelism through ForkUnion thread pools and custom "executors".
- ✅ [Semantic Search](#usearch--uform--ucall--multimodal-semantic-search) and approximate products & attention for LLM context.

[faiss]: https://github.com/facebookresearch/faiss
[usearch-header]: https://github.com/unum-cloud/USearch/blob/main/include/usearch/dense/ram_index.hpp
[obscure-use-cases]: https://ashvardanian.com/posts/abusing-vector-search
[simd]: https://en.wikipedia.org/wiki/Single_instruction,_multiple_data
[faster-than-faiss]: https://www.unum.cloud/blog/2023-11-07-scaling-vector-search-with-intel
[clickhouse-docs]: https://clickhouse.com/docs/en/engines/table-engines/mergetree-family/annindexes#usearch
[duckdb-docs]: https://duckdb.org/2024/05/03/vector-similarity-search-vss.html

__Technical Insights__ and related articles:

- [Uses Arm SVE and x86 AVX-512's masked loads to eliminate tail `for`-loops](https://ashvardanian.com/posts/simsimd-faster-scipy/#tails-of-the-past-the-significance-of-masked-loads).
- [Uses Horner's method for polynomial approximations, beating GCC 12 by 119×](https://ashvardanian.com/posts/gcc-12-vs-avx512fp16/).
- [For every language implements a custom separate binding](https://ashvardanian.com/posts/porting-cpp-library-to-ten-languages/).

## Comparison with FAISS

FAISS is a widely recognized standard for high-performance vector search engines.
USearch and FAISS both employ the same HNSW algorithm, but they differ significantly in their design principles.
USearch is compact and broadly compatible without sacrificing performance, primarily focusing on user-defined metrics and fewer dependencies.

|                                              |                   FAISS |                  USearch |             Improvement |
| :------------------------------------------- | ----------------------: | -----------------------: | ----------------------: |
| Indexing time ⁰                              |                         |                          |                         |
| 100 Million 96d `f32`, `f16`, `i8` vectors   |       2.6 · 2.6 · 2.6 h |        0.3 · 0.2 · 0.2 h | __9.6 · 10.4 · 10.7 x__ |
| 100 Million 1536d `f32`, `f16`, `i8` vectors |       5.0 · 4.1 · 3.8 h |        2.1 · 1.1 · 0.8 h |   __2.3 · 3.6 · 4.4 x__ |
|                                              |                         |                          |                         |
| Codebase length ¹                            |       84 K [SLOC][sloc] |         3 K [SLOC][sloc] |            maintainable |
| Supported metrics ²                          |         9 fixed metrics |               any metric |              extendible |
| Supported languages ³                        |             C++, Python |             12 languages |                portable |
| Supported ID types ⁴                         |          32-bit, 64-bit |                  128-bit |               efficient |
| Filtering ⁵                                  |               ban-lists |           any predicates |              composable |
| Required dependencies ⁶                      |            BLAS, OpenMP |                        - |            light-weight |
| Bindings ⁷                                   |                    SWIG |                   Native |             low-latency |
| Python binding size ⁸                        | [~ 10 MB][faiss-weight] | [< 1 MB][usearch-weight] |              deployable |

[sloc]: https://en.wikipedia.org/wiki/Source_lines_of_code
[faiss-weight]: https://pypi.org/project/faiss-cpu/#files
[usearch-weight]: https://pypi.org/project/usearch/#files

> ⁰ [Tested][intel-benchmarks] on Intel Sapphire Rapids, with the simplest inner-product distance, equivalent recall, and memory consumption while also providing far superior search speed.
> ¹ A shorter codebase of `usearch/` over `faiss/` makes the project easier to maintain and audit.
> ² User-defined metrics allow you to customize your search for various applications, from GIS to creating custom metrics for composite embeddings from multiple AI models or hybrid full-text and semantic search.
> ³ With USearch, you can reuse the same preconstructed index in various programming languages.
> ⁴ Keys are 128 bits wide and need not be unique, so a UUID is an identity rather than something to map, and several rows may share one key - which is what late interaction needs.
> ⁵ With USearch the index can be combined with arbitrary external containers, like Bloom filters or third-party databases, to filter out irrelevant keys during index traversal.
> ⁶ Lack of obligatory dependencies makes USearch much more portable.
> ⁷ Native bindings introduce lower call latencies than more straightforward approaches.
> ⁸ Lighter bindings make downloads and deployments faster.

[intel-benchmarks]: https://www.unum.cloud/blog/2023-11-07-scaling-vector-search-with-intel

Base functionality is identical to FAISS, and the interface must be familiar if you have ever investigated Approximate Nearest Neighbors search:

```py
# pip install usearch

import numpy as np
from usearch import DenseIndex

index = DenseIndex(dimensions=3)            # Default settings for 3D vectors
vectors = np.array([[0.2, 0.6, 0.4]], dtype=np.float32)
index.reserve(1)                            # Freeze the geometry and size the arenas
index.add([42], vectors)                    # Every verb is a batch; one vector is one row
matches = index.search(vectors, 10)         # Find 10 nearest neighbors per query row

assert int(matches[0][0].key) == 42
assert matches[0][0].distance <= 0.001
assert np.allclose(index[42], vectors, atol=0.1) # High tolerance for mixed-precision comparisons
```

More settings are always available, and the API is designed to be as flexible as possible.
The default storage/quantization level is hardware-dependant for efficiency, but `bf16` is recommended for most modern CPUs.

```py
index = DenseIndex(
    dimensions=3, # Scalars per vector; 0 defers the geometry to a later `load`
    metric='angular', # Or 'inner', 'euclidean', 'hamming', 'jaccard'
    quantization='bf16', # Store as 'f64', 'f32', 'bf16', 'f16', 'e5m2', 'e4m3', 'e3m2', 'e2m3', 'i8', 'u8', 'u1x8'
    connectivity=16, # Optional: Limit number of neighbors per graph node
    residency='ram', # Optional: Or 'vram' on a GPU, 'nvme' for external memory
)
```

The candidate-pool breadth rides each `add` and `search` call as `expansion_factor` rather than the handle, so one immutable index serves a whole recall sweep.
Keys are 128 bits wide and need not be unique, so `multi` has no counterpart either.

## Serialization & Serving An Index from Disk

USearch checkpoints ride the [SafeGraphs](https://github.com/safegraphs/SafeGraphs) interchange format, in two forms:

- Into a __monolith__, one `.safegraphs` file or a buffer of `storage_requirements` bytes, holding every shard as one graph.
- Into a __manifest__, a `.safegraphs.json` catalog beside one monolith per shard, for billion-scale indexes spread over drives.

Either form loads into any shard count, and an external-memory residency opens a monolith in place, enabling you to optimize your server choices for indexing speed and serving costs.
This can result in __20x cost reduction__ on AWS and other public clouds.

```py
index.save("index.safegraphs")
index.load("index.safegraphs")

# A checkpoint's geometry without opening a shard; the keys are the constructor's own.
reader = DenseIndex(**DenseIndex.metadata("index.safegraphs"))
reader.load("index.safegraphs")

# Serving from external memory is a residency, not a separate verb.
external = DenseIndex(residency="nvme")
external.load("index.safegraphs")
```

Where the payload lives decides what loading means: materialize into RAM, upload to a device, or open in place on disk.
An external-memory handle is read-only, and bigger indexes come from `merge` rather than from further insertion.

## Exact vs. Approximate Search

Approximate search methods, such as HNSW, are predominantly used when an exact brute-force search becomes too resource-intensive.
This typically occurs when you have millions of entries in a collection.
For smaller collections, reach for [NumKong](https://github.com/ashvardanian/numkong) directly:
its `cdist` is the same SIMD kernel family the index dispatches to, without a graph in the way.

```py
import numkong as nk
import numpy as np

vectors = np.random.rand(10_000, 1024).astype(np.float32)
queries = np.random.rand(10, 1024).astype(np.float32)
distances = nk.cdist(queries, vectors, "angular")
```

When compared to FAISS's `IndexFlatL2` in Google Colab, __[those kernels may offer up to a 20× performance improvement](https://github.com/unum-cloud/USearch/issues/176#issuecomment-1666650778)__:

- `faiss.IndexFlatL2`: __55.3 ms__.
- NumKong's `cdist`: __2.54 ms__.

USearch itself reads the same walk as arithmetic through `multiply_into`, an approximate $A \times B^{T}$ that reports every pair the traversal scored, and `attend_into`, FlashAttention folded into that walk for LLM context.

## Exact vs. Approximate Attention

Attention is a GEMM, a softmax and another GEMM, and `attend` runs all three inside one graph walk.
The walk scores keys against the query, which is the first GEMM, but only for the keys it reaches.
Every scored key enters an online softmax with a running maximum and partition, and its value row joins the weighted sum the moment it is scored, which is the second GEMM.
So the long tail of keys the walk visits on its way counts too, not only the best matches, and an exhaustive walk is exact attention.

- __Values stay with the caller.__
  A lookup resolves each scored key to its value row in the caller's KV cache, as wide as the keys and stored like them.
- __One index per key-value head.__
  The query heads of a grouped-query attention group share one walk per token: it ranks nodes by the best head's logit and reads every key and value row once for all of them.
- __Recent tokens stay out of the graph.__
  A token is inserted once it leaves the dense window, so the window and the graph never count a key twice, and their outputs merge through the softmax carry, the maximum and partition of each head.

[NumKong](https://github.com/ashvardanian/numkong)'s attention is the exact counterpart, and the two meet at that carry:

- __Work.__
  NumKong scores every key of a segment, streaming a packed KV cache in panels, so its cost grows with the context.
  USearch scores the keys its walk reaches, so its cost grows with the walk, not with the context.
- __Layout.__
  NumKong first packs keys and values into a per-ISA layout.
  USearch keeps the keys in its graph and reads values in place through the lookup.
- __Shapes.__
  NumKong covers ragged segments, any grouped-query ratio, causal and sliding-window bands, and head depths up to 256 on its SIMD paths, with wider heads on its serial kernel.
  USearch takes any depth the index froze and any number of query heads sharing its key-value head, but no masks: causality comes from the publication watermark, and the window is whatever has not been inserted yet.
- __Use.__
  Dense attention fits the recent window and short contexts; the graph fits the far context of million-token windows; one partial-softmax merge combines both.

## One Modality per Domain

Rather than one index taking an arbitrary distance function, each modality carries the measures its domain actually needs, with SIMD kernels dispatched by the machine's capabilities:

| Modality   | Header                       | Measures                                                |
| :--------- | :--------------------------- | :------------------------------------------------------ |
| Dense      | `usearch/dense_index.h`      | Inner, Angular, Euclidean, Hamming, Jaccard             |
| Sequence   | `usearch/sequence_index.h`   | Levenshtein, Needleman-Wunsch, Smith-Waterman           |
| Matrix     | `usearch/matrix_index.h`     | Chamfer, Hausdorff, Gromov-Wasserstein, Kabsch, Umeyama |
| Geospatial | `usearch/geospatial_index.h` | Karney geodesics, trajectory and boundary similarity    |

Moreover, unlike older approaches indexing high-dimensional spaces, like KD-Trees and Locality Sensitive Hashing, HNSW doesn't require vectors to be identical in length.
They only have to be comparable.
So you can apply it in [obscure][obscure] applications, like searching for similar sets or fuzzy text matching, using [GZip][gzip-similarity] compression-ratio as a distance function.

[obscure]: https://ashvardanian.com/posts/abusing-vector-search
[gzip-similarity]: https://twitter.com/LukeGessler/status/1679211291292889100?s=20

## Filtering and Predicate Functions

Sometimes you may want to cross-reference search-results against some external database or filter them based on some criteria.
In most engines, you'd have to manually perform paging requests, successively filtering the results.
In USearch you can simply pass a predicate function to the search method, which will be applied directly during graph traversal.
In Rust that would look like this:

```rust
let is_odd = |key: Key| key % 2 == 1;
let query = vec![0.2, 0.1, 0.2, 0.1, 0.3];
let results = index.filtered_search(&query, 10, is_odd).unwrap();
assert!(
    results.keys.iter().all(|&key| key % 2 == 1),
    "All keys must be odd"
);
```

## Memory Efficiency, Downcasting, and Quantization

Training a quantization model and dimension-reduction is a common approach to accelerate vector search.
Those, however, are only sometimes reliable, can significantly affect the statistical properties of your data, and require regular adjustments if your distribution shifts.
Instead, we have focused on high-precision arithmetic over low-precision downcasted vectors.
The same index, and `add` and `search` operations will automatically down-cast or up-cast between `f64_t`, `f32_t`, `bf16_t`, `f16_t`, `e5m2_t`, `e4m3_t`, `e3m2_t`, `e2m3_t`, `i8_t`, `u8_t`, the packed nibbles `i4x2_t` and `u4x2_t`, and single-bit `u1x8_t` representations.
You can use the following command to check, if hardware acceleration is enabled:

```sh
$ python -c 'from usearch import DenseIndex; print(DenseIndex(dimensions=768, metric="angular", quantization="f16").capabilities)'
> sapphire
$ python -c 'from usearch import DenseIndex; print(DenseIndex(dimensions=166, metric="jaccard").capabilities)'
> ice
```

In most cases, `bf16` is recommended for modern CPUs.
For even smaller footprints, USearch supports IEEE & MX-compatible Float8 (`e5m2` and `e4m3`) and Float6 (`e3m2` and `e2m3`) formats.
You can pass pre-quantized buffers from [NumKong](https://github.com/ashvardanian/numkong) with the explicit `dtype=` parameter on `add` and `search`, or let USearch handle the quantization internally from higher-precision inputs.
When quantization is enabled, the "get"-like functions won't be able to recover the original data, so you may want to replicate the original vectors elsewhere.
When quantizing to `i8_t` integers, note that it's only valid for cosine-like metrics.
As part of the quantization process, the vectors are normalized to unit length and later scaled to [-127, 127] range to occupy the full 8-bit range.
When quantizing to `u1x8_t` single-bit representations, note that it's only valid for binary metrics like Jaccard, Hamming, etc.
As part of the quantization process, the scalar components greater than zero are set to `true`, and the rest to `false`.

![USearch uint40_t support](https://github.com/unum-cloud/USearch/blob/main/assets/usearch-neighbor-types.png?raw=true)

Using smaller numeric types will save you RAM needed to store the vectors, but you can also compress the neighbors lists forming our proximity graphs.
By default, 32-bit `uint32_t` is used to enumerate those, which is not enough if you need to address over 4 Billion entries.
For such cases we provide a custom `uint40_t` type, that will still be 37.5% more space-efficient than the commonly used 8-byte integers, and will scale up to 1 Trillion entries.

## Functionality

Every verb crosses one C interface, so a binding either exposes it or has not been asked to yet.
Kotlin and Scala consume the Java surface directly and match its row.

|                             | C++ | C 99 | Python | Rust | Go  | Java | JS  | Swift | C#  | Zig | Mojo | Wolfram |
| :-------------------------- | :-: | :--: | :----: | :--: | :-: | :--: | :-: | :---: | :-: | :-: | :--: | :-----: |
| Add, search, remove         | ✅  |  ✅  |   ✅   |  ✅  | ✅  |  ✅  | ✅  |  ✅   | ✅  | ✅  |  ✅  |   ✅    |
| Save, load, metadata        | ✅  |  ✅  |   ✅   |  ✅  | ✅  |  ✅  | ✅  |  ✅   | ✅  | ✅  |  ✅  |   ✅    |
| Batch operations            | ✅  |  ✅  |   ✅   |  ✅  | ✅  |  ✅  | ✅  |  ✅   | ✅  | ✅  |  ✅  |   ✅    |
| Residency: RAM, VRAM, NVMe  | ✅  |  ✅  |   ✅   |  ✅  | ✅  |  ✅  | ✅  |  ✅   | ✅  | ✅  |  ✅  |   ✅    |
| Filter predicates           | ✅  |  ✅  |   ✅   |  ✅  | ✅  |  ❌  | ❌  |  ✅   | ✅  | ✅  |  ✅  |   ❌    |
| Refinement                  | ✅  |  ✅  |   ✅   |  ✅  | ✅  |  ✅  | ✅  |  ✅   | ✅  | ✅  |  ✅  |   ❌    |
| Late interaction: `maxsim`  | ✅  |  ✅  |   ✅   |  ✅  | ✅  |  ✅  | ✅  |  ✅   | ✅  | ✅  |  ✅  |   ❌    |
| Sparse products: `multiply` | ✅  |  ✅  |   ✅   |  ✅  | ❌  |  ❌  | ✅  |  ✅   | ❌  | ✅  |  ✅  |   ❌    |
| Attention: `attend`         | ✅  |  ✅  |   ✅   |  ✅  | ❌  |  ❌  | ✅  |  ✅   | ❌  | ✅  |  ✅  |   ❌    |
| Streaming cursors           | ✅  |  ✅  |   ✅   |  ✅  | ❌  |  ❌  | ✅  |  ✅   | ✅  | ✅  |  ✅  |   ❌    |
| Resumable `find`            | ✅  |  ✅  |   ✅   |  ✅  | ✅  |  ✅  | ✅  |  ✅   | ✅  | ✅  |  ✅  |   ❌    |
| Index merging               | ✅  |  ✅  |   ✅   |  ✅  | ❌  |  ❌  | ❌  |  ❌   | ❌  | ❌  |  ❌  |   ❌    |

## Application Examples

### USearch + UForm + UCall = Multimodal Semantic Search

AI has a growing number of applications, but one of the coolest classic ideas is to use it for Semantic Search.
One can take an encoder model, like the multi-modal [UForm](https://github.com/unum-cloud/uform), and a web-programming framework, like [UCall](https://github.com/unum-cloud/ucall), and build a text-to-image search platform in just 20 lines of Python.

```python
from ucall import Server
from uform import get_model, Modality
from usearch import DenseIndex

import numpy as np
import PIL as pil

processors, models = get_model('unum-cloud/uform3-image-text-english-small')
model_text = models[Modality.TEXT_ENCODER]
model_image = models[Modality.IMAGE_ENCODER]
processor_text = processors[Modality.TEXT_ENCODER]
processor_image = processors[Modality.IMAGE_ENCODER]

server = Server()
index = DenseIndex(dimensions=256)
index.reserve(1_000_000)

@server
def add(key: int, photo: pil.Image.Image):
    image = processor_image(photo)
    vector = model_image(image)
    index.add([key], vector.reshape(1, -1))

@server
def search(query: str) -> np.ndarray:
    tokens = processor_text(query)
    vector = model_text(tokens)
    matches = index.search(vector.reshape(1, -1), 3)
    return np.asarray(matches.keys)

server.run()
```

Similar experiences can also be implemented in other languages and on the client side, removing the network latency.
For Swift and iOS, check out the [`ashvardanian/SwiftSemanticSearch`](https://github.com/ashvardanian/SwiftSemanticSearch) repository.

<table>
  <tr>
    <td>
      <img src="https://github.com/ashvardanian/ashvardanian/blob/master/demos/SwiftSemanticSearch-Dog.gif?raw=true" alt="SwiftSemanticSearch demo Dog">
    </td>
    <td>
      <img src="https://github.com/ashvardanian/ashvardanian/blob/master/demos/SwiftSemanticSearch-Flowers.gif?raw=true" alt="SwiftSemanticSearch demo with Flowers">
    </td>
  </tr>
</table>

A more complete [demo with Streamlit is available on GitHub](https://github.com/ashvardanian/usearch-images).
We have pre-processed some commonly used datasets, cleaned the images, produced the vectors, and pre-built the index.

| Dataset                             |            Modalities | Images |                              Download |
| :---------------------------------- | --------------------: | -----: | ------------------------------------: |
| [Unsplash][unsplash-25k-origin]     | Images & Descriptions |   25 K | [HuggingFace / Unum][unsplash-25k-hf] |
| [Conceptual Captions][cc-3m-origin] | Images & Descriptions |    3 M |        [HuggingFace / Unum][cc-3m-hf] |
| [Arxiv][arxiv-2m-origin]            |    Titles & Abstracts |    2 M |     [HuggingFace / Unum][arxiv-2m-hf] |

[unsplash-25k-origin]: https://github.com/unsplash/datasets
[cc-3m-origin]: https://huggingface.co/datasets/conceptual_captions
[arxiv-2m-origin]: https://www.kaggle.com/datasets/Cornell-University/arxiv

[unsplash-25k-hf]: https://huggingface.co/datasets/unum-cloud/ann-unsplash-25k
[cc-3m-hf]: https://huggingface.co/datasets/unum-cloud/ann-cc-3m
[arxiv-2m-hf]: https://huggingface.co/datasets/unum-cloud/ann-arxiv-2m

### USearch + RDKit = Molecular Search

Comparing molecule graphs and searching for similar structures is expensive and slow.
It can be seen as a special case of the NP-Complete Subgraph Isomorphism problem.
Luckily, domain-specific approximate methods exist.
The one commonly used in Chemistry is to generate structures from [SMILES][smiles] and later hash them into binary fingerprints.
The latter are searchable with binary similarity metrics, like the Tanimoto coefficient.
Below is an example using the RDKit package.

```python
from usearch import DenseIndex
from rdkit import Chem
from rdkit.Chem import AllChem

import numpy as np

molecules = [Chem.MolFromSmiles('CCOC'), Chem.MolFromSmiles('CCO')]
encoder = AllChem.GetRDKitFPGenerator()

fingerprints = np.vstack([encoder.GetFingerprint(x) for x in molecules])
fingerprints = np.packbits(fingerprints, axis=1)

index = DenseIndex(dimensions=2048, metric="jaccard", quantization="u1x8")
index.reserve(len(molecules))
keys = np.arange(len(molecules))

index.add(keys, fingerprints)
matches = index.search(fingerprints, 10)
```

That method was used to build the ["USearch Molecules"](https://github.com/unum-science/USearchMolecules), one of the largest Chem-Informatics datasets, containing 7 billion small molecules and 28 billion fingerprints.

[smiles]: https://en.wikipedia.org/wiki/Simplified_molecular-input_line-entry_system
[rdkit-fingerprints]: https://www.rdkit.org/docs/RDKit_Book.html#additional-information-about-the-fingerprints

### USearch + POI Coordinates = GIS Applications

Space and time get their own modality rather than a user-supplied distance function.
`geospatial_index.h` carries Karney geodesics - which account for the Earth's oblateness, where the Haversine formula does not - alongside trajectory and boundary similarities.

```py
from usearch import GeospatialIndex

index = GeospatialIndex(layout="elevated_timed")
index.add(keys, trajectories)
matches = index.search(trajectory, 10, mode="route")
```

The layout names which axes a position carries - latitude and longitude, optionally altitude and a timestamp - and the mode names what a comparison means, from nearest approach between two point sets to the shape of a whole route.
A tolerance supplies the exchange rate between metres and seconds per call, so the same index answers both a tight spatial query and a loose temporal one.

## Integrations & Users

- [x] ClickHouse: [C++](https://github.com/ClickHouse/ClickHouse/pull/53447), [docs](https://clickhouse.com/docs/en/engines/table-engines/mergetree-family/annindexes#usearch).
- [x] DuckDB: [post](https://duckdb.org/2024/05/03/vector-similarity-search-vss.html).
- [x] ScyllaDB: [Rust](https://github.com/scylladb/vector-store), [presentation](https://www.slideshare.net/slideshow/vector-search-with-scylladb-by-szymon-wasik/276571548).
- [x] TiDB & TiFlash: [C++](https://github.com/pingcap/tiflash), [announcement](https://www.pingcap.com/article/introduce-vector-search-indexes-in-tidb/).
- [x] YugaByte: [C++](https://github.com/yugabyte/yugabyte-db/blob/366b9f5e3c4df3a1a17d553db41d6dc50146f488/src/yb/vector_index/usearch_wrapper.cc).
- [x] MemGraph: [C++](https://github.com/memgraph/memgraph/blob/784dd8520f65050d033aea8b29446e84e487d091/src/storage/v2/indices/vector_index.cpp), [announcement](https://memgraph.com/blog/simplify-data-retrieval-memgraph-vector-search).
- [x] Google: [UniSim](https://github.com/google/unisim), [RetSim](https://arxiv.org/abs/2311.17264) paper.
- [x] LangChain: [Python](https://github.com/langchain-ai/langchain/releases/tag/v0.0.257) and [JavaScript](https://github.com/langchain-ai/langchainjs/releases/tag/0.0.125).
- [x] Microsoft Semantic Kernel: [Python](https://github.com/microsoft/semantic-kernel/releases/tag/python-0.3.9.dev) and C#.
- [x] GPTCache: [Python](https://github.com/zilliztech/GPTCache/releases/tag/0.1.29).
- [x] Sentence-Transformers: Python [docs](https://www.sbert.net/docs/package_reference/quantization.html#sentence_transformers.quantization.semantic_search_usearch).
- [x] Pathway: [Rust](https://github.com/pathwaycom/pathway).
- [x] Vald: [GoLang](https://github.com/vdaas/vald).
- [x] MatrixOne: [GoLang](https://github.com/matrixorigin/matrixone).

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
