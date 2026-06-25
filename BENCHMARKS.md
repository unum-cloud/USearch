# Benchmarking USearch

For reproducible recall-vs-throughput sweeps against alternative search engines across the full quantization matrix, use the [RetriEval](https://github.com/ashvardanian/RetriEval) suite.
The in-tree scripts described below are kept for internal R&D and fine-grained profiling.

## Hyper-parameters

All major HNSW implementation share an identical list of hyper-parameters:

- connectivity (often called `M`),
- expansion on additions (often called `efConstruction`),
- expansion on search (often called `ef`).

The default values vary drastically.

| Library   | Connectivity | EF @ A | EF @ S |
| :-------- | -----------: | -----: | -----: |
| `hnswlib` |           16 |    200 |     10 |
| `FAISS`   |           32 |     40 |     16 |
| `USearch` |           16 |    128 |     64 |

Below are the performance numbers for a benchmark running on the 64 cores of AWS `c7g.metal` "Graviton 3"-based instances.
The main columns are:

- __Add__: Number of insertion Queries Per Second.
- __Search__: Number search Queries Per Second.
- __Recall @1__: How often does approximate search yield the exact best match?

### Different "connectivity"

| Vectors    | Connectivity | EF @ A | EF @ S | __Add__, QPS | __Search__, QPS | __Recall @1__ |
| :--------- | -----------: | -----: | -----: | -----------: | --------------: | ------------: |
| `f32` x256 |           16 |    128 |     64 |       75'640 |         131'654 |         99.3% |
| `f32` x256 |           12 |    128 |     64 |       81'747 |         149'728 |         99.0% |
| `f32` x256 |           32 |    128 |     64 |       64'368 |         104'050 |         99.4% |

### Different "expansion factors"

| Vectors    | Connectivity | EF @ A | EF @ S | __Add__, QPS | __Search__, QPS | __Recall @1__ |
| :--------- | -----------: | -----: | -----: | -----------: | --------------: | ------------: |
| `f32` x256 |           16 |    128 |     64 |       75'640 |         131'654 |         99.3% |
| `f32` x256 |           16 |     64 |     32 |      128'644 |         228'422 |         97.2% |
| `f32` x256 |           16 |    256 |    128 |       39'981 |          69'065 |         99.2% |

### Different vectors "quantization"

| Vectors      | Connectivity | EF @ A | EF @ S | __Add__, QPS | __Search__, QPS | __Recall @1__ |
| :----------- | -----------: | -----: | -----: | -----------: | --------------: | ------------: |
| `f32` x256   |           16 |    128 |     64 |       87'995 |         171'856 |         99.1% |
| `f16` x256   |           16 |    128 |     64 |       87'270 |         153'788 |         98.4% |
| `f16` x256 ✳️ |           16 |    128 |     64 |       71'454 |         132'673 |         98.4% |
| `i8` x256    |           16 |    128 |     64 |      115'923 |         274'653 |         98.9% |

As seen on the chart, for `f16` quantization, performance may differ depending on native hardware support for that numeric type.
Also worth noting, 8-bit quantization results in almost no quantization loss and may perform better than `f16`.

## Utilities

Within this repository you will find two commonly used utilities:

- `cpp/bench.cpp` the produces the `bench_cpp` binary for broad USearch benchmarks.
- `python/bench.py` and `python/bench.ipynb` for interactive charts against FAISS.

### C++ Benchmarking Utilities

To achieve best highest results we suggest compiling locally for the target architecture.

```sh
git submodule update --init --recursive
cmake -DUSEARCH_BUILD_BENCH_CPP=1 -DUSEARCH_BUILD_TEST_C=1 -DUSEARCH_USE_NUMKONG=1 -DUSEARCH_USE_OPENMP=1 -DCMAKE_BUILD_TYPE=RelWithDebInfo -B build_profile
cmake --build build_profile --config RelWithDebInfo --parallel
build_profile/bench_cpp --help
```

Which would print the following instructions.

```txt
SYNOPSIS
        build_profile/bench_cpp [--vectors <path>] [--queries <path>] [--neighbors <path>] [-o
                                <path>] [-b] [-j <integer>] [-c <integer>] [--expansion-add
                                <integer>] [--expansion-search <integer>] [--rows-skip <integer>]
                                [--rows-take <integer>] [--dtype <type>] [--metric <name>] [-h]

OPTIONS
        --vectors <path>
                    .[fhbd]bin, .i8bin, .u8bin, .f32bin file path to construct the index

        --queries <path>
                    .[fhbd]bin, .i8bin, .u8bin, .f32bin file path to query the index

        --neighbors <path>
                    .ibin file path with ground truth

        -o, --output <path>
                    .usearch output file path

        -b, --big   Will switch to uint40_t for neighbors lists with over 4B entries
        -j, --threads <integer>
                    Uses all available cores by default

        -c, --connectivity <integer>
                    Index granularity

        --expansion-add <integer>
                    Affects indexing depth

        --expansion-search <integer>
                    Affects search depth

        --rows-skip <integer>
                    Number of vectors to skip

        --rows-take <integer>
                    Number of vectors to take

        --dtype <type>
                    Quantization type: f64, f32, bf16, f16, e5m2, e4m3, e3m2, e2m3, i8, u8, b1

        --metric <name>
                    Distance metric: ip, l2sq, cos, hamming, tanimoto, sorensen, haversine

        -h, --help  Print this help information on this tool and exit
```

Here is an example of running the C++ benchmark:

```sh
build_profile/bench_cpp \
    --vectors datasets/wiki_1M/base.1M.fbin \
    --queries datasets/wiki_1M/query.public.100K.fbin \
    --neighbors datasets/wiki_1M/groundtruth.public.100K.ibin \
    --dtype bf16 --metric ip

build_profile/bench_cpp \
    --vectors datasets/t2i_1B/base.1B.fbin \
    --queries datasets/t2i_1B/query.public.100K.fbin \
    --neighbors datasets/t2i_1B/groundtruth.public.100K.ibin \
    --output datasets/t2i_1B/index.usearch \
    --dtype bf16 --metric cos
```

> Optional parameters include `connectivity`, `expansion_add`, `expansion_search`.

For Python, jut open the Jupyter Notebook and start playing around.

### Python Benchmarking Utilities

Several benchmarking suites are available for Python: approximate search, exact search, and clustering.

```sh
python/scripts/bench.py --help
python/scripts/bench_exact.py --help
python/scripts/bench_cluster.py --help
```

These are intended as smoke tests and ad-hoc profiling helpers.
Published recall-vs-throughput numbers should be reproduced with
[RetriEval](https://github.com/ashvardanian/RetriEval), which covers
the full `b1` / `i8` / `u8` / `e2m3` / `e3m2` / `e4m3` / `e5m2` / `f16`
/ `bf16` / `f32` / `f64` matrix on standard datasets.

## Datasets

BigANN benchmark is a good starting point, if you are searching for large collections of high-dimensional vectors.
Those often come with precomputed ground-truth neighbors, which is handy for recall evaluation.
Datasets below are grouped by scale; only configurations with matching ground truth support recall evaluation.

### ~1M Scale — Development & Testing

| Dataset                                    | Scalar Type | Dimensions | Metric | Base Size |    Ground Truth     |
| :----------------------------------------- | :---------: | :--------: | :----: | :-------: | :-----------------: |
| [Unum UForm Wiki][unum-wiki-1m]            |    `f32`    |    256     |   IP   |   1 GB    |  100K queries, yes  |
| [Unum UForm Creative Captions][unum-cc-3m] |    `f32`    |    256     |   IP   |   3 GB    | cross-modal pairing |
| [Arxiv with E5][unum-arxiv-2m]             |    `f32`    |    768     |   IP   |   6 GB    | cross-modal pairing |

### ~10M Scale

| Dataset                              | Scalar Type | Dimensions | Metric | Base Size |   Ground Truth    |
| :----------------------------------- | :---------: | :--------: | :----: | :-------: | :---------------: |
| [Meta BIGANN (SIFT)][bigann]         |    `u8`     |    128     |   L2   |  1.2 GB   | 10K queries, yes  |
| [Microsoft Turing-ANNS][msft-turing] |    `f32`    |    100     |   L2   |  3.7 GB   | 100K queries, yes |
| [Yandex Deep][yandex-deep]           |    `f32`    |     96     |   L2   |  3.6 GB   |  ¹ no subset GT   |

> ¹ Yandex only publishes ground truth computed against the full 1B dataset. A `base.10M.fbin` exists for
> download but using 1B ground truth with a subset will produce misleadingly low recall. Use it only for
> throughput/latency testing, not recall evaluation.

### ~100M Scale

| Dataset                              | Scalar Type | Dimensions | Metric | Base Size |   Ground Truth    |
| :----------------------------------- | :---------: | :--------: | :----: | :-------: | :---------------: |
| [Meta BIGANN (SIFT)][bigann]         |    `u8`     |    128     |   L2   |   12 GB   | 10K queries, yes  |
| [Microsoft Turing-ANNS][msft-turing] |    `f32`    |    100     |   L2   |   37 GB   | 100K queries, yes |
| [Microsoft SpaceV][msft-spacev]      |    `i8`     |    100     |   L2   |  9.3 GB   | 30K queries, yes  |

### ~1B Scale

| Dataset                              | Scalar Type | Dimensions | Metric | Base Size |   Ground Truth    |
| :----------------------------------- | :---------: | :--------: | :----: | :-------: | :---------------: |
| [Meta BIGANN (SIFT)][bigann]         |    `u8`     |    128     |   L2   |  119 GB   | 10K queries, yes  |
| [Microsoft Turing-ANNS][msft-turing] |    `f32`    |    100     |   L2   |  373 GB   | 100K queries, yes |
| [Microsoft SpaceV][msft-spacev]      |    `i8`     |    100     |   L2   |   93 GB   | 30K queries, yes  |
| [Yandex Text-to-Image][yandex-t2i]   |    `f32`    |    200     |  Cos   |  750 GB   | 100K queries, yes |
| [Yandex Deep][yandex-deep]           |    `f32`    |     96     |   L2   |  358 GB   | 10K queries, yes  |
|                                      |             |            |        |           |                   |
| [ViT-L/12 LAION][laion]              |    `f32`    |    2048    |  Cos   | 2 - 10 TB |         —         |

[unum-cc-3m]: https://huggingface.co/datasets/unum-cloud/ann-cc-3m
[unum-wiki-1m]: https://huggingface.co/datasets/unum-cloud/ann-wiki-1m
[unum-arxiv-2m]: https://huggingface.co/datasets/unum-cloud/ann-arxiv-2m
[msft-spacev]: https://github.com/ashvardanian/SpaceV
[msft-turing]: https://learning2hash.github.io/publications/microsoftturinganns1B/
[yandex-t2i]: https://research.yandex.com/blog/benchmarks-for-billion-scale-similarity-search
[yandex-deep]: https://research.yandex.com/blog/benchmarks-for-billion-scale-similarity-search
[bigann]: https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/bigann/
[laion]: https://laion.ai/blog/laion-5b/#download-the-data

### Unum UForm Creative Captions

A cross-modal dataset of ~2.75M image-text pairs embedded with UForm VL English (256 dimensions).
No separate query/ground-truth files — the natural ground truth is the image-text pairing: `image[i]` matches `text[i]`.

```sh
mkdir -p datasets/cc_3M/ && \
    wget -nc https://huggingface.co/datasets/unum-cloud/ann-cc-3m/resolve/main/images.uform-vl-english.fbin \
        -O datasets/cc_3M/images.fbin && \
    wget -nc https://huggingface.co/datasets/unum-cloud/ann-cc-3m/resolve/main/texts.uform-vl-english.fbin \
        -O datasets/cc_3M/texts.fbin
```

To benchmark cross-modal join:

```bash
python python/scripts/join.py \
    --vectors-a datasets/cc_3M/texts.fbin \
    --vectors-b datasets/cc_3M/images.fbin \
    --metric cos --diagnostics
```

### Unum UForm Wiki

```sh
mkdir -p datasets/wiki_1M/ && \
    wget -nc https://huggingface.co/datasets/unum-cloud/ann-wiki-1m/resolve/main/base.1M.fbin -P datasets/wiki_1M/ &&
    wget -nc https://huggingface.co/datasets/unum-cloud/ann-wiki-1m/resolve/main/query.public.100K.fbin -P datasets/wiki_1M/ &&
    wget -nc https://huggingface.co/datasets/unum-cloud/ann-wiki-1m/resolve/main/groundtruth.public.100K.ibin -P datasets/wiki_1M/
```

### Arxiv with E5

```sh
mkdir -p datasets/arxiv_2M/ && \
    wget -nc https://huggingface.co/datasets/unum-cloud/ann-arxiv-2m/resolve/main/abstract.e5-base-v2.fbin -P datasets/arxiv_2M/ &&
    wget -nc https://huggingface.co/datasets/unum-cloud/ann-arxiv-2m/resolve/main/title.e5-base-v2.fbin -P datasets/arxiv_2M/
```

### Yandex Text-to-Image

> __Warning:__ Yandex only publishes ground truth computed against the full 1B dataset.
> A `base.1M.fbin` subset exists for download but has no matching ground truth — using 1B ground truth
> with the 1M subset will produce misleadingly low recall. Use `base.1M.fbin` only for throughput/latency
> testing, not recall evaluation.

```sh
mkdir -p datasets/t2i_1B/ && \
    wget -nc https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/base.1B.fbin -P datasets/t2i_1B/ &&
    wget -nc https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/query.public.100K.fbin -P datasets/t2i_1B/ &&
    wget -nc https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/groundtruth.public.100K.ibin -P datasets/t2i_1B/
```

To run the benchmark (requires the full 1B base for valid recall):

```bash
build_profile/bench_cpp \
    --vectors datasets/t2i_1B/base.1B.fbin \
    --queries datasets/t2i_1B/query.public.100K.fbin \
    --neighbors datasets/t2i_1B/groundtruth.public.100K.ibin \
    --output datasets/t2i_1B/index.usearch \
    --dtype bf16 --metric cos
```

### Yandex Deep

> __Warning:__ Yandex only publishes ground truth computed against the full 1B dataset.
> Smaller base files (`base.10M.fbin`, `base.1M.fbin`) are available for download but have no matching
> ground truth — using 1B ground truth with a subset will produce misleadingly low recall. Use those files
> only for throughput/latency testing, not recall evaluation.

```sh
mkdir -p datasets/deep_1B/ && \
    wget -nc https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/base.1B.fbin -P datasets/deep_1B/ &&
    wget -nc https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/query.public.10K.fbin -P datasets/deep_1B/ &&
    wget -nc https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/groundtruth.public.10K.ibin -P datasets/deep_1B/
```

To run the benchmark (requires the full 1B base for valid recall):

```bash
build_profile/bench_cpp \
    --vectors datasets/deep_1B/base.1B.fbin \
    --queries datasets/deep_1B/query.public.10K.fbin \
    --neighbors datasets/deep_1B/groundtruth.public.10K.ibin \
    --output datasets/deep_1B/index.usearch \
    --dtype bf16 --metric l2sq
```

### Meta BIGANN — SIFT

The full 1B dataset is available from Meta. No pre-sliced subset base files exist, so range requests are
used to download only the first N vectors, followed by a header patch to update the vector count.
Pre-computed ground truth is available for 10M and 100M subsets.

#### 10M subset, ~1.2 GB

```sh
mkdir -p datasets/sift_10M/ && \
    wget -nc https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/bigann/query.public.10K.u8bin -P datasets/sift_10M/ && \
    wget -nc https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/GT_10M/bigann-10M -O datasets/sift_10M/groundtruth.public.10K.ibin && \
    wget --header="Range: bytes=0-1280000007" \
        https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/bigann/base.1B.u8bin \
        -O datasets/sift_10M/base.10M.u8bin && \
    python3 -c "
import struct
with open('datasets/sift_10M/base.10M.u8bin', 'r+b') as f:
    f.write(struct.pack('I', 10_000_000))
"
```

```bash
build_profile/bench_cpp \
    --vectors datasets/sift_10M/base.10M.u8bin \
    --queries datasets/sift_10M/query.public.10K.u8bin \
    --neighbors datasets/sift_10M/groundtruth.public.10K.ibin \
    --output datasets/sift_10M/index.usearch \
    --dtype u8 --metric l2sq
```

#### 100M subset, ~12 GB

```sh
mkdir -p datasets/sift_100M/ && \
    wget -nc https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/bigann/query.public.10K.u8bin -P datasets/sift_100M/ && \
    wget -nc https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/GT_100M/bigann-100M -O datasets/sift_100M/groundtruth.public.10K.ibin && \
    wget --header="Range: bytes=0-12800000007" \
        https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/bigann/base.1B.u8bin \
        -O datasets/sift_100M/base.100M.u8bin && \
    python3 -c "
import struct
with open('datasets/sift_100M/base.100M.u8bin', 'r+b') as f:
    f.write(struct.pack('I', 100_000_000))
"
```

```bash
build_profile/bench_cpp \
    --vectors datasets/sift_100M/base.100M.u8bin \
    --queries datasets/sift_100M/query.public.10K.u8bin \
    --neighbors datasets/sift_100M/groundtruth.public.10K.ibin \
    --output datasets/sift_100M/index.usearch \
    --dtype u8 --metric l2sq
```

### Microsoft Turing-ANNS

The full 1B dataset is ~373 GB of `f32` vectors with 100 dimensions.
Subsets can be obtained via range requests, followed by a header patch to update the vector count.
Pre-computed ground truth is available for 1M, 10M, and 100M subsets.

#### 1M subset, ~400 MB

```sh
mkdir -p datasets/turing_1M/ && \
    wget -nc https://comp21storage.z5.web.core.windows.net/comp21/MSFT-TURING-ANNS/query100K.fbin \
        -O datasets/turing_1M/query.public.100K.fbin && \
    wget -nc https://comp21storage.z5.web.core.windows.net/comp21/MSFT-TURING-ANNS/msturing-gt-1M \
        -O datasets/turing_1M/groundtruth.public.100K.ibin && \
    wget --header="Range: bytes=0-400000007" \
        https://comp21storage.z5.web.core.windows.net/comp21/MSFT-TURING-ANNS/base1b.fbin \
        -O datasets/turing_1M/base.1M.fbin && \
    python3 -c "
import struct
with open('datasets/turing_1M/base.1M.fbin', 'r+b') as f:
    f.write(struct.pack('I', 1_000_000))
"
```

```bash
build_profile/bench_cpp \
    --vectors datasets/turing_1M/base.1M.fbin \
    --queries datasets/turing_1M/query.public.100K.fbin \
    --neighbors datasets/turing_1M/groundtruth.public.100K.ibin \
    --output datasets/turing_1M/index.usearch \
    --dtype bf16 --metric l2sq
```

#### 10M subset, ~3.7 GB

```sh
mkdir -p datasets/turing_10M/ && \
    wget -nc https://comp21storage.z5.web.core.windows.net/comp21/MSFT-TURING-ANNS/query100K.fbin \
        -O datasets/turing_10M/query.public.100K.fbin && \
    wget -nc https://comp21storage.z5.web.core.windows.net/comp21/MSFT-TURING-ANNS/msturing-gt-10M \
        -O datasets/turing_10M/groundtruth.public.100K.ibin && \
    wget --header="Range: bytes=0-4000000007" \
        https://comp21storage.z5.web.core.windows.net/comp21/MSFT-TURING-ANNS/base1b.fbin \
        -O datasets/turing_10M/base.10M.fbin && \
    python3 -c "
import struct
with open('datasets/turing_10M/base.10M.fbin', 'r+b') as f:
    f.write(struct.pack('I', 10_000_000))
"
```

```bash
build_profile/bench_cpp \
    --vectors datasets/turing_10M/base.10M.fbin \
    --queries datasets/turing_10M/query.public.100K.fbin \
    --neighbors datasets/turing_10M/groundtruth.public.100K.ibin \
    --output datasets/turing_10M/index.usearch \
    --dtype bf16 --metric l2sq
```

#### 100M subset, ~37 GB

```sh
mkdir -p datasets/turing_100M/ && \
    wget -nc https://comp21storage.z5.web.core.windows.net/comp21/MSFT-TURING-ANNS/query100K.fbin \
        -O datasets/turing_100M/query.public.100K.fbin && \
    wget -nc https://comp21storage.z5.web.core.windows.net/comp21/MSFT-TURING-ANNS/msturing-gt-100M \
        -O datasets/turing_100M/groundtruth.public.100K.ibin && \
    wget --header="Range: bytes=0-40000000007" \
        https://comp21storage.z5.web.core.windows.net/comp21/MSFT-TURING-ANNS/base1b.fbin \
        -O datasets/turing_100M/base.100M.fbin && \
    python3 -c "
import struct
with open('datasets/turing_100M/base.100M.fbin', 'r+b') as f:
    f.write(struct.pack('I', 100_000_000))
"
```

```bash
build_profile/bench_cpp \
    --vectors datasets/turing_100M/base.100M.fbin \
    --queries datasets/turing_100M/query.public.100K.fbin \
    --neighbors datasets/turing_100M/groundtruth.public.100K.ibin \
    --output datasets/turing_100M/index.usearch \
    --dtype bf16 --metric l2sq
```

### Microsoft SpaceV

The original dataset can be pulled in a USearch-compatible form from AWS S3:

```sh
mkdir -p datasets/spacev_1B/ && \
    aws s3 cp s3://your-bucket/path/to/spacev/ datasets/spacev_1B/ --recursive
```

A smaller 100M dataset can be pulled from Hugging Face.

```sh
mkdir -p datasets/spacev_100M/ && \
    wget -nc https://huggingface.co/datasets/unum-cloud/ann-spacev-100m/resolve/main/ids.100M.i32bin -P datasets/spacev_100M/ &&
    wget -nc https://huggingface.co/datasets/unum-cloud/ann-spacev-100m/resolve/main/base.100M.i8bin -P datasets/spacev_100M/ &&
    wget -nc https://huggingface.co/datasets/unum-cloud/ann-spacev-100m/resolve/main/query.30K.i8bin -P datasets/spacev_100M/ &&
    wget -nc https://huggingface.co/datasets/unum-cloud/ann-spacev-100m/resolve/main/groundtruth.30K.i32bin -P datasets/spacev_100M/ &&
    wget -nc https://huggingface.co/datasets/unum-cloud/ann-spacev-100m/resolve/main/groundtruth.30K.f32bin -P datasets/spacev_100M/
```

```bash
build_profile/bench_cpp \
    --vectors datasets/spacev_100M/base.100M.i8bin \
    --queries datasets/spacev_100M/query.30K.i8bin \
    --neighbors datasets/spacev_100M/groundtruth.30K.i32bin \
    --output datasets/spacev_100M/index.usearch \
    --dtype i8 \
    --metric l2sq
```

## Profiling

With `perf`:

```sh
# Pass environment variables with `-E`, and `-d` for details
sudo -E perf stat -d build_profile/bench_cpp ...
sudo -E perf mem -d build_profile/bench_cpp ...
# Sample on-CPU functions for the specified command, at 1 Kilo Hertz:
sudo -E perf record -F 1000 build_profile/bench_cpp ...
perf record -d -e arm_spe// -- build_profile/bench_cpp ..
```

### Caches

```sh
sudo perf stat -e 'faults,dTLB-loads,dTLB-load-misses,cache-misses,cache-references' build_profile/bench_cpp ...
```

Typical output on a 1M vectors dataset is:

```txt
            255426      faults                                                      
      305988813388      dTLB-loads                                                  
        8845723783      dTLB-load-misses          #    2.89% of all dTLB cache accesses
       20094264206      cache-misses              #    6.567 % of all cache refs    
      305988812745      cache-references                                            

       8.285148010 seconds time elapsed

     500.705967000 seconds user
       1.371118000 seconds sys
```

If you notice problems and the stalls are closer to 90%, it might be a good reason to consider enabling Huge Pages and tuning allocations alignment.
To enable Huge Pages:

```sh
sudo cat /proc/sys/vm/nr_hugepages
sudo sysctl -w vm.nr_hugepages=2048
sudo reboot
sudo cat /proc/sys/vm/nr_hugepages
```
