# Compiling USearch

## Build

### Linux

```
cmake -B ./build_release -DCMAKE_BUILD_TYPE=Release -DUSEARCH_USE_OPENMP=1 -DUSEARCH_USE_JEMALLOC=0 && make -C ./build_release -j

cmake -B ./build_release -DCMAKE_BUILD_TYPE=Release -DUSEARCH_USE_OPENMP=1 -DUSEARCH_USE_JEMALLOC=1 && make -C ./build_release -j

./build_release/bench \
    --vectors datasets/wiki_1M/base.1M.fbin \
    --queries datasets/wiki_1M/query.public.100K.fbin \
    --neighbors datasets/wiki_1M/groundtruth.public.100K.ibin \
    --ip
```

### MacOS

```
brew install libomp llvm
cmake \
    -DCMAKE_C_COMPILER="/opt/homebrew/opt/llvm/bin/clang" \
    -DCMAKE_CXX_COMPILER="/opt/homebrew/opt/llvm/bin/clang++" \
     -B ./build_release && \
    make -C ./build_release -j

./build_release/bench \
    --vectors datasets/wiki_1M/base.1M.fbin \
    --queries datasets/wiki_1M/query.public.100K.fbin \
    --neighbors datasets/wiki_1M/groundtruth.public.100K.ibin \
    --ip

mv build_release/usearch.cpython* src/ && \
    python src/bench.py \
    --vectors datasets/wiki_1M/base.1M.fbin \
    --queries datasets/wiki_1M/query.public.100K.fbin \
    --neighbors datasets/wiki_1M/groundtruth.public.100K.ibin \
    --ip    
```

### Python

```sh
pip install -e .
```

### JavaScript

```sh
npm install
node src/test.js
```

### JavaScript

```sh
cargo run
```

## Benchmarks

### Datasets

### Evaluation

Yandex Text-to-Image embeddings.

```sh
export path_vectors=datasets/t2i_1M/base.1M.fbin
export path_queries=datasets/t2i_1M/query.public.100K.fbin
export path_neighbors=datasets/t2i_1M/groundtruth.public.100K.ibin
./build_release/bench
```

Wikipedia UForm embeddings.

```sh
export path_vectors=datasets/wiki_1M/base.1M.fbin
export path_queries=datasets/wiki_1M/query.public.100K.fbin 
export path_neighbors=datasets/wiki_1M/groundtruth.public.100K.ibin
./build_release/bench
```

### Profiling

With `perf`:

```sh
# Pass environment variables with `-E`, and `-d` for details
sudo -E perf stat -d ./build_release/bench
sudo -E perf mem -d ./build_release/bench
# Sample on-CPU functions for the specified command, at 1 Kilo Hertz:
sudo -E perf record -F 1000 ./build_release/bench
perf record -d -e arm_spe// -- ./build_release/bench
```
