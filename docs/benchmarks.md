# Benchmarking USearch

## Hyper-parameters

All major HNSW implementation share an identical list of hyper-parameters:

- connectivity (often called `M`),
- expansion on additions (often called `efConstruction`),
- expansion on search (often called `ef`).

The default values vary drastically.

|  Library  | Connectivity | EF @ A | EF @ S |
| :-------: | :----------: | :----: | :----: |
| `hnswlib` |      16      |  200   |   10   |
|  `FAISS`  |      32      |   40   |   16   |
| `USearch` |      16      |  128   |   64   |

Below are the performance numbers for a benchmark running on the 64 cores of AWS `c7g.metal` "Graviton 3"-based instances.
The main columns are:

- __Add__: Number of insertion Queries Per Second.
- __Search__: Number search Queries Per Second.
- __Recall @1__: How often does approximate search yield the exact best match?

### Different "connectivity"

| Vectors    | Connectivity | EF @ A | EF @ S | __Add__, QPS | __Search__, QPS | __Recall @1__ |
| :--------- | :----------: | :----: | :----: | :----------: | :-------------: | ------------: |
| `f32` x256 |      16      |  128   |   64   |    75'640    |     131'654     |         99.3% |
| `f32` x256 |      12      |  128   |   64   |    81'747    |     149'728     |         99.0% |
| `f32` x256 |      32      |  128   |   64   |    64'368    |     104'050     |         99.4% |

### Different "expansion factors"

| Vectors    | Connectivity | EF @ A | EF @ S | __Add__, QPS | __Search__, QPS | __Recall @1__ |
| :--------- | :----------: | :----: | :----: | :----------: | :-------------: | ------------: |
| `f32` x256 |      16      |  128   |   64   |    75'640    |     131'654     |         99.3% |
| `f32` x256 |      16      |   64   |   32   |   128'644    |     228'422     |         97.2% |
| `f32` x256 |      16      |  256   |  128   |    39'981    |     69'065      |         99.2% |

### Different vectors "quantization"

| Vectors      | Connectivity | EF @ A | EF @ S | __Add__, QPS | __Search__, QPS | __Recall @1__ |
| :----------- | :----------: | :----: | :----: | :----------: | :-------------: | ------------: |
| `f32` x256   |      16      |  128   |   64   |    87'995    |     171'856     |         99.1% |
| `f16` x256   |      16      |  128   |   64   |    87'270    |     153'788     |         98.4% |
| `f16` x256 ✳️ |      16      |  128   |   64   |    71'454    |     132'673     |         98.4% |
| `i8` x256    |      16      |  128   |   64   |   115'923    |     274'653     |         98.9% |

As seen on the chart, for `f16` quantization, performance may differ depending on native hardware support for that numeric type.
Also worth noting, 8-bit quantization results in almost no quantization loss and may perform better than `f16`.

## Utilities

Within this repository you will find two commonly used utilities:

- `cpp/bench.cpp` the produces the `bench` binary for broad USearch benchmarks.
- `python/bench.py` and `python/bench.ipynb` for interactive charts against FAISS.

To achieve best highest results we suggest compiling locally for the target architecture.

```sh
cmake -B ./build_release \
    -DCMAKE_BUILD_TYPE=Release \
    -DUSEARCH_USE_OPENMP=1 \
    -DUSEARCH_USE_SIMSIMD=1 \
    -DUSEARCH_USE_FP16LIB=0 \
    -DUSEARCH_USE_JEMALLOC=1 && \
    make -C ./build_release -j

./build_release/bench --help
```

Which would print the following instructions.

```txt
SYNOPSIS
        ./build_release/bench [--vectors <path>] [--queries <path>] [--neighbors <path>] [-b] [-j
                              <integer>] [-c <integer>] [--expansion-add <integer>]
                              [--expansion-search <integer>] [--native|--f16quant|--i8quant]
                              [--ip|--l2sq|--cos|--haversine] [-h]

OPTIONS
        --vectors <path>
                    .fbin file path to construct the index

        --queries <path>
                    .fbin file path to query the index

        --neighbors <path>
                    .ibin file path with ground truth

        -b, --big   Will switch to uint40_t for neighbors lists with over 4B entries
        -j, --threads <integer>
                    Uses all available cores by default

        -c, --connectivity <integer>
                    Index granularity

        --expansion-add <integer>
                    Affects indexing depth

        --expansion-search <integer>
                    Affects search depth

        --native    Use raw templates instead of type-punned classes
        --f16quant  Enable `f16_t` quantization
        --i8quant   Enable `int8_t` quantization
        --ip        Choose Inner Product metric
        --l2sq        Choose L2 Euclidean metric
        --cos       Choose Angular metric
        --haversine Choose Haversine metric
        -h, --help  Print this help information on this tool and exit
```

Here is an example of running the C++ benchmark:

```sh
./build_release/bench \
    --vectors datasets/wiki_1M/base.1M.fbin \
    --queries datasets/wiki_1M/query.public.100K.fbin \
    --neighbors datasets/wiki_1M/groundtruth.public.100K.ibin

./build_release/bench \
    --vectors datasets/t2i_1B/base.1B.fbin \
    --queries datasets/t2i_1B/query.public.100K.fbin \
    --neighbors datasets/t2i_1B/groundtruth.public.100K.ibin \
    --output datasets/t2i_1B/index.usearch \
    --cos
```


> Optional parameters include `connectivity`, `expansion_add`, `expansion_search`.

For Python, jut open the Jupyter Notebook and start playing around.

## Datasets

BigANN benchmark is a good starting point, if you are searching for large collections of high-dimensional vectors.
Those often come with precomputed ground-truth neighbors, which is handy for recall evaluation.

| Dataset                                    | Scalar Type | Dimensions | Metric |   Size    |
| :----------------------------------------- | :---------: | :--------: | :----: | :-------: |
| [Unum UForm Creative Captions][unum-cc-3m] |   float32   |    256     |   IP   |   3 GB    |
| [Unum UForm Wiki][unum-wiki-1m]            |   float32   |    256     |   IP   |   1 GB    |
| [Yandex Text-to-Image Sample][unum-t2i]    |   float32   |    200     |  Cos   |   1 GB    |
|                                            |             |            |        |           |
| [Microsoft SPACEV][spacev]                 |    int8     |    100     |   L2   |   93 GB   |
| [Microsoft Turing-ANNS][turing]            |   float32   |    100     |   L2   |  373 GB   |
| [Yandex Deep1B][deep]                      |   float32   |     96     |   L2   |  358 GB   |
| [Yandex Text-to-Image][t2i]                |   float32   |    200     |  Cos   |  750 GB   |
|                                            |             |            |        |           |
| [ViT-L/12 LAION][laion]                    |   float32   |    2048    |  Cos   | 2 - 10 TB |

Luckily, smaller samples of those datasets are available.

[unum-cc-3m]: https://huggingface.co/datasets/unum-cloud/ann-cc-3m
[unum-wiki-1m]: https://huggingface.co/datasets/unum-cloud/ann-wiki-1m
[unum-t2i]: https://huggingface.co/datasets/unum-cloud/ann-t2i-1m
[spacev]: https://github.com/microsoft/SPTAG/tree/main/datasets/SPACEV1B
[turing]: https://learning2hash.github.io/publications/microsoftturinganns1B/
[t2i]: https://research.yandex.com/blog/benchmarks-for-billion-scale-similarity-search
[deep]: https://research.yandex.com/blog/benchmarks-for-billion-scale-similarity-search
[laion]: https://laion.ai/blog/laion-5b/#download-the-data

### Unum UForm Wiki

```sh
mkdir -p datasets/wiki_1M/ && \
    wget -nc https://huggingface.co/datasets/unum-cloud/ann-wiki-1m/resolve/main/base.1M.fbin -P datasets/wiki_1M/ &&
    wget -nc https://huggingface.co/datasets/unum-cloud/ann-wiki-1m/resolve/main/query.public.100K.fbin -P datasets/wiki_1M/ &&
    wget -nc https://huggingface.co/datasets/unum-cloud/ann-wiki-1m/resolve/main/groundtruth.public.100K.ibin -P datasets/wiki_1M/
```

### Yandex Text-to-Image

```sh
mkdir -p datasets/t2i_1B/ && \
    wget -nc https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/base.1B.fbin -P datasets/t2i_1B/ &&
    wget -nc https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/base.1M.fbin -P datasets/t2i_1B/ &&
    wget -nc https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/query.public.100K.fbin -P datasets/t2i_1B/ &&
    wget -nc https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/groundtruth.public.100K.ibin -P datasets/t2i_1B/
```

### Yandex Deep1B

```sh
mkdir -p datasets/deep_1B/ && \
    wget -nc https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/base.1B.fbin -P datasets/deep_1B/ &&
    wget -nc https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/base.10M.fbin -P datasets/deep_1B/ &&
    wget -nc https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/query.public.10K.fbin -P datasets/deep_1B/ &&
    wget -nc https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/groundtruth.public.10K.ibin -P datasets/deep_1B/
```

### Arxiv with E5

```sh
mkdir -p datasets/arxiv_2M/ && \
    wget -nc https://huggingface.co/datasets/unum-cloud/ann-arxiv-2m/resolve/main/abstract.e5-base-v2.fbin -P datasets/arxiv_2M/ &&
    wget -nc https://huggingface.co/datasets/unum-cloud/ann-arxiv-2m/resolve/main/title.e5-base-v2.fbin -P datasets/arxiv_2M/
```

## Profiling

With `perf`:

```sh
# Pass environment variables with `-E`, and `-d` for details
sudo -E perf stat -d ./build_release/bench ...
sudo -E perf mem -d ./build_release/bench ...
# Sample on-CPU functions for the specified command, at 1 Kilo Hertz:
sudo -E perf record -F 1000 ./build_release/bench ...
perf record -d -e arm_spe// -- ./build_release/bench ..
```

### Caches

```sh
sudo perf stat -e 'faults,dTLB-loads,dTLB-load-misses,cache-misses,cache-references' ./build_release/bench ...
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

