# USearch for C

## Installation

USearch vector-search engine can be integrated into your project using CMake.
Alternatively, you can download one of the precompiled binaries from the [releases page](https://github.com/unum-cloud/usearch/releases).

## Quickstart

```c
#include <stdio.h> // For printing.
#include <assert.h> // For assertions!
#include <usearch/usearch.h> // For the win 🚀

int main() {

    // Construct:
    size_t dimensions = 128;
    usearch_error_t error = NULL;
    usearch_init_options_t opts = {
        .metric_kind = usearch_metric_cos_k,
        .scalar_kind = usearch_scalar_f16_k,
        .dimensions = dimensions,
        .expansion_add = 0, // for defaults
        .expansion_search = 0 // for defaults
    };
    usearch_index_t index = usearch_init(&opts, &error);

    size_t vectors_count = 1000;
    usearch_reserve(index, vectors_count, &error);
    if (error) goto cleanup;

    // Populate:
    float vector[dimensions]; // don't forget to fill the vector with data
    usearch_add(index, 42, &vector[0], usearch_scalar_f32_k, &error);
    if (error) goto cleanup;
    
    // Check up:
    assert(usearch_size(index, &error) == 1);
    assert(usearch_capacity(index, &error) == vectors_count);
    assert(usearch_contains(index, 42, &error));

    // Search:
    usearch_key_t found_keys[10];
    usearch_distance_t found_distances[10];
    size_t found_count = usearch_search(
        index, &vector[0], usearch_scalar_f32_k, 10, 
        &found_keys[0], &found_distances[0], &error);

  cleanup:
    if (error) fprintf(stderr, "Error: %s\n", error);
    if (index) usearch_free(index, &error);
    return error ? 1 : 0;
}
```

## Serialization

To save and load the index from disk, use the following methods:

```c
usearch_save(index, "index.usearch", &error);
usearch_load(index, "index.usearch", &error);
usearch_view(index, "index.usearch", &error);
```

Before saving to disk, you can determine exactly how much disk space the index will take:

```c
size_t bytes = usearch_serialized_length(index, &error);
```

With a known size, you can serialize the index into an in-memory buffer:

```c
void* buffer = malloc(bytes);
usearch_save_buffer(index, buffer, bytes, &error);
usearch_load_buffer(index, buffer, bytes, &error);
usearch_view_buffer(index, buffer, bytes, &error);
```

One can also retrieve index metadata from the buffer or from disk:

```c
usearch_init_options_t opts;
usearch_metadata("index.usearch", &opts, &error);
usearch_metadata_buffer(buffer, bytes, &opts, &error);
```

## Metrics

USearch comes pre-packaged with SimSIMD, bringing over 100 SIMD-accelerated distance kernels for x86 and ARM architectures.
That includes:

- `usearch_metric_cos_k` - Cosine Similarity metric, defined as `Cos = 1 - sum(a[i] * b[i]) / (sqrt(sum(a[i]^2) * sqrt(sum(b[i]^2)))`.
- `usearch_metric_ip_k` - Inner Product metric, defined as `IP = 1 - sum(a[i] * b[i])`.
- `usearch_metric_l2sq_k` - Squared Euclidean Distance metric, defined as `L2 = sum((a[i] - b[i])^2)`.
- `usearch_metric_haversine_k` - Haversine (Great Circle) Distance metric.
- `usearch_metric_divergence_k` - Jensen Shannon Divergence metric.
- `usearch_metric_pearson_k` - Pearson Correlation metric.
- `usearch_metric_hamming_k` - Bit-level Hamming Distance metric, defined as the number of differing bits.
- `usearch_metric_tanimoto_k` - Bit-level Tanimoto (Jaccard) metric, defined as the number of intersecting bits divided by the number of union bits.
- `usearch_metric_sorensen_k` - Bit-level Sorensen metric.

### User-Defined Metrics

You can also define your own metrics by implementing the `usearch_metric_t` interface:

```c
simsimd_distance_t callback(void const* a, void const* b, void* state) {
    // Your custom metric implementation here
}

void callback_state = NULL;
usearch_change_metric(index, callback, callback_state, usearch_metric_unknown_k, &error);
```

You can always revert back to one of the native metrics by calling:

```c
usearch_change_metric_kind(index, usearch_metric_cos_k, &error);
```

## Filtering with Predicates

Sometimes you may want to cross-reference search-results against some external database or filter them based on some criteria.
In most engines, you'd have to manually perform paging requests, successively filtering the results.
In USearch you can simply pass a predicate function to the search method, which will be applied directly during graph traversal.

```c
int is_odd(usearch_key_t key, void* state) {
    return key % 2;
}

usearch_key_t found_keys[10];
usearch_distance_t found_distances[10];
usearch_filtered_search(
    index, &query[0], usearch_scalar_f32_k, 10, 
    &is_odd, NULL, // no state needed for this callback
    &found_keys[0], &found_distances[0], &error);
```

## Extracting, Updating, and Removing Values

It is generally not recommended to use HNSW indexes in case of frequent removals or major distribution shifts.
For small updates, you can use the following methods:

```c
float recovered_vector[dimensions];
size_t count_retrieved = usearch_get(index, 42, 1, 
    &recovered_vector[0], usearch_scalar_f32_k, &error);
assert(count_retrieved <= 1);

size_t count_renamed = usearch_rename(index, 42, 43, &error);
size_t count_removed = usearch_remove(index, 43, &error);
assert(count_renamed == count_removed);
```

USearch can also be used for multi-indices, where each key may map to multiple vectors.
That's a common case when implementing semantic search for long documents, where the entire documents can't fit into one input sequence in a transformer-model, and you have to split them into chunks.

```c
float many_vectors[10][dimensions];
size_t count_retrieved = usearch_get(index, 42, 10, 
    &many_vectors[0][0], usearch_scalar_f32_k, &error);
assert(count_retrieved <= 10);
```

## Exact Search

USearch exposes its internal SIMD-accelerated distance functions for exact search.
To invoke them for a pair of vectors, use the following methods:

```c
float vector_a[dimensions], vector_b[dimensions];
usearch_distance_t distance = usearch_distance(
    &vector_a[0], &vector_b[0], usearch_scalar_f32_k, dimensions, usearch_metric_cos_k, &error);
```

Alternatively, you can benefit from faster thread-pools and priority queues for parallel exact batch-search.

```c
size_t threads = 0;
size_t top_k = 10;
size_t dataset_count = 1000, queries_count = 10;
simsimd_f16_t dataset[dataset_count][dimensions];
simsimd_f16_t queries[queries_count][dimensions];

usearch_key_t resulting_keys[queries_count][top_k];
usearch_distance_t resulting_distances[queries_count][top_k];

usearch_exact_search(
    &dataset[0][0], dataset_count, dimensions * sizeof(simsimd_f16_t),
    &queries[0][0], queries_count, dimensions * sizeof(simsimd_f16_t),
    usearch_scalar_f16_k, top_k, threads,
    &resulting_keys[0][0], sizeof(usearch_key_t) * top_k,
    &resulting_distances[0][0], sizeof(usearch_distance_t) * top_k,
    &error);
```

## Concurrency and Parallelism

USearch manages a pool of "thread local" data-structures to avoid contention and improve performance.
By default, as many of these structures are created as there are logical cores on the machine.
Then, one can construct the index in parallel using many cores using third-party thread-pools or tools like OpenACC and OpenMP.

```c
usearch_change_threads_add(index, omp_get_max_threads(), &error);
usearch_change_threads_search(index, omp_get_max_threads(), &error);

#pragma omp parallel for
for (size_t i = 0; i < 1000; i++) {
    usearch_add(index, i, &vector[0], usearch_scalar_f32_k, &error);
}

#pragma omp parallel for
for (size_t i = 0; i < 1000; i++) {
    usearch_key_t found_keys[10];
    usearch_distance_t found_distances[10];
    size_t found_matches = usearch_search(index, 
        &vector[0], usearch_scalar_f32_k, 10, 
        &found_keys[0], &found_distances[0], &error);
}
```

## Performance Tuning

To optimize the performance of the index, you can adjust the expansion values used during index creation and search operations.
Higher expansion values will lead to better search accuracy at the cost of slightly increased memory usage, but potentially much higher search times.
Following methods are available to adjust the expansion values:

```c
printf("Connectivity: %zu\n", usearch_connectivity(index, &error)); // can't change
printf("Add expansion: %zu\n", usearch_expansion_add(index, &error)); // can change
printf("Search expansion: %zu\n", usearch_expansion_search(index, &error)); // can change
usearch_change_expansion_add(index, 32, &error);
usearch_change_expansion_search(index, 32, &error);
```

Optimizing hardware utilization, you may want to check the SIMD hardware acceleration capabilities of the index and memory consumption.
The first will print the codename of the most advanced SIMD instruction set supported by the CPU and used by the index.
The second will print the memory usage of the index in bytes.

```c
printf("Hardware acceleration: %s\n", usearch_hardware_acceleration(index, &error));
printf("Memory usage: %zu\n", usearch_memory_usage(index, &error));
```
