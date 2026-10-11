# USearch for C

## Installation

USearch vector-search engine can be integrated into your project using CMake.
Alternatively, you can download one of the precompiled binaries from the [releases page](https://github.com/unum-cloud/USearch/releases).
One header, `usearch/usearch.h`, pulls in every modality: `dense_index.h`, `sequence_index.h`, `matrix_index.h` and `geospatial_index.h`.

## Quickstart

Every verb takes one argument struct, initialized by designator, so every field left out keeps its default.
Every call reports through an `outcome`, and every call blocks unless its `completion` says `usearch_completion_deferred_k`, so the outcome is final by the time the call returns.

```c
#include <stdio.h>               // For printing.
#include <usearch/usearch.h>     // For the win 🚀

int main(void) {
    usearch_outcome_t outcome = {0};
    usearch_executor_t executor = NULL;
    usearch_arena_t arena = NULL;
    usearch_dense_index_t index = NULL;
    usearch_key_t keys[2] = {{{1}}, {{2}}};
    float vectors[2][128] = {{0}};
    usearch_matches_view_t matches = {0};

    // Threads live in the executor, the data lives in the index, and outputs land in the arena.
    usearch_executor_for_ram(&(usearch_executor_for_ram_t){.executor = &executor, .outcome = &outcome});
    if (outcome.status != usearch_status_success_k) goto cleanup;
    usearch_arena_init(&(usearch_arena_init_t){.arena = &arena, .executor = executor, .outcome = &outcome});
    if (outcome.status != usearch_status_success_k) goto cleanup;
    usearch_dense_init_ram(&(usearch_dense_init_ram_t){
        .index = &index, .outcome = &outcome,
        .metric = usearch_dense_metric_angular_k, .quantization = usearch_scalar_f16_k, .dimensions = 128});
    if (outcome.status != usearch_status_success_k) goto cleanup;

    // Populate:
    usearch_dense_insert(&(usearch_dense_insert_t){
        .index = index, .executor = executor, .outcome = &outcome,
        .count = 2, .keys = {keys, sizeof(usearch_key_t)},
        .vectors = {vectors, sizeof(vectors[0]), 128, usearch_scalar_f32_k}});
    if (outcome.status != usearch_status_success_k) goto cleanup;

    // Search one query; a zero stride would repeat it across a larger `count`.
    usearch_dense_search(&(usearch_dense_search_t){
        .index = index, .executor = executor, .outcome = &outcome,
        .count = 1, .queries = {vectors[0], 0, 128, usearch_scalar_f32_k},
        .matches = &matches, .arena = arena, .wanted = 10});
    if (outcome.status != usearch_status_success_k) goto cleanup;
    printf("Found %zu matches\n", matches.counts[0]);

cleanup:
    if (outcome.message) fprintf(stderr, "%s: %s\n", usearch_status_name(outcome.status), outcome.message);
    usearch_dense_free(index);
    usearch_arena_free(arena);
    usearch_executor_free(executor);
    return outcome.status == usearch_status_success_k ? 0 : 1;
}
```

Keys are 16 bytes wide, enough for a UUID, a content hash, or a 64-bit integer with room to spare.
Messages are static strings, never freed, and `NULL` exactly on success.

## Executors and Asynchrony

An executor owns the threads or device streams that run calls, and the traversal scratch they walk with.
One executor serves any number of indexes, and calls on one executor start in submission order.

```c
int32_t const sockets[2] = {0, 1};
usearch_executor_for_ram(&(usearch_executor_for_ram_t){
    .executor = &executor, .outcome = &outcome,
    .ram = {sockets, 2}, // NUMA nodes whose cores it spawns on
    .threads = 0});      // zero takes 7/8 of those cores
```

An application with its own ForkUnion pool lends it instead with `usearch_executor_on_pool`, and GPUs get `usearch_executor_for_vram` with the device ordinals and, optionally, the streams to run on.
A `NULL` executor is accepted by every verb and runs the call on the caller's own thread, final when it returns whatever its `completion`, which suits size lookups and other short calls.

A deferred call returns once submitted and keeps using what its argument struct points to until its outcome is final, so many calls can be deferred at once, each with its own outcome and its own arena, which it holds until then:

```c
usearch_matches_view_t matches[4];
usearch_outcome_t outcomes[4] = {{0}};
for (size_t shard = 0; shard != 4; ++shard)
    usearch_dense_search(&(usearch_dense_search_t){
        .index = shards[shard], .executor = executor, .outcome = &outcomes[shard],
        .completion = usearch_completion_deferred_k,
        .count = 32, .queries = {queries, 128 * sizeof(float), 128, usearch_scalar_f32_k},
        .matches = &matches[shard], .arena = arenas[shard], .wanted = 10});
for (size_t shard = 0; shard != 4; ++shard) usearch_outcome_wait(&outcomes[shard]);
```

The status reads `usearch_status_deferred_k` until the call is final, then the library stores the message and the final status with release ordering, and touches nothing the struct names after that; the caller reads it with acquire ordering, by waiting or polling.
Each executor tracks its deferred calls in a fixed table of `deferred_limit` slots.
A deferred submission that finds the table full finishes at once with `usearch_status_contended_k`, so submitting never allocates, and blocking calls take no slot at all.
A blocking call on an executor runs after every earlier call finishes and before any later one starts.
`usearch_executor_drain` waits for everything submitted before it, then sets the executor's idle mood: `usearch_mood_parked_k` lets idle threads sleep, `usearch_mood_spinning_k` keeps them hot.
Every `*_free` waits until each deferred call naming its handle is final, so a handle may be dropped at any moment.

Long calls take a progress handler, which is also the only way to stop one:

```c
usearch_continuation_t stop_when_asked(size_t completed, size_t total, void* state) {
    return *(int const*)state ? usearch_cancel_k : usearch_continue_k;
}
```

A stopped call reports `usearch_status_cancelled_k`.
An insertion claims every row before its first report, so its handler only reports, and all its rows publish at once.

## Retrieving Vectors and Arenas

Every output whose length scales with the call lands in an arena: the matches of a search, every vector stored under one key, every pair a sparse product scored, a monolith, a cursor's batch.
An arena is memory the library allocates and the caller passes to call after call, each of which rewinds it and writes views into it, growing only when the new outputs outsize everything it held.
An executor makes it where it computes, and a `NULL` executor makes host pages for calls on the calling thread; it may outlive the executor that made it.
Only fixed-size reports, such as stats, descriptions and `written` counts, stay fields the caller points at.

```c
usearch_arena_t arena = NULL;
usearch_arena_init(&(usearch_arena_init_t){.arena = &arena, .executor = executor, .outcome = &outcome});
for (size_t i = 0; i != 2; ++i) {
    usearch_rows_view_t rows = {0};
    size_t rows_count = 0;
    usearch_dense_get(&(usearch_dense_get_t){
        .index = index, .executor = executor, .outcome = &outcome, .key = keys[i],
        .vectors = &rows, .written = &rows_count, .arena = arena, .outputs_scalar = usearch_scalar_f32_k});
    if (outcome.status != usearch_status_success_k) break;
    // `rows.data` holds `rows_count` vectors `rows.stride` bytes apart, valid until the next call given `arena`
}
usearch_arena_free(arena);
```

Any verb may reuse any arena of its residency, so a search and a get can share one, as long as the first call's views are not read after the second starts.
To keep one call's outputs alive past the next, pass the next call another arena.
An arena serves one call at a time: a blocking call borrows it, a deferred call holds it until final, and a second call naming it meanwhile finishes `usearch_status_contended_k`.
A `NULL` arena, or inputs pointing into the arena a call is about to rewind, refuse with `usearch_status_invalid_argument_k`, and a call that fails leaves its outputs `NULL`.

## GPUs

A VRAM index lives on the devices its `vram` options name, one shard per device, and a VRAM executor drives it through one stream per device.
Passing your own streams lets USearch work join the rest of your pipeline:

```c
#include <cuda_runtime.h>

cudaStream_t stream;
cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
int32_t const device = 0;
usearch_executor_t gpu = NULL;
usearch_executor_for_vram(&(usearch_executor_for_vram_t){
    .executor = &gpu, .outcome = &outcome, .vram = {&device, 1}, .streams = (void* const*)&stream});

usearch_dense_index_t index = NULL;
usearch_dense_init_vram(&(usearch_dense_init_vram_t){
    .index = &index, .outcome = &outcome, .metric = usearch_dense_metric_inner_k,
    .quantization = usearch_scalar_bf16_k, .dimensions = 1024, .vram = {&device, 1}});
usearch_dense_reserve(&(usearch_dense_reserve_t){.index = index, .executor = gpu, .outcome = &outcome, .capacity = 1000000});
```

The library classifies every pointer it is given through the driver, so inputs may live anywhere:
rows already on the device are read in place, pinned host rows upload directly, and only pageable host rows pass through the executor's pinned bounce buffer.
Pin the rows you upload often, and the bounce disappears:

```c
float* rows = NULL;
cudaHostAlloc((void**)&rows, 4096 * 1024 * sizeof(float), cudaHostAllocDefault);
// ... fill 4096 rows ...
usearch_dense_insert(&(usearch_dense_insert_t){
    .index = index, .executor = gpu, .outcome = &outcome,
    .count = 4096, .keys = {row_keys, sizeof(usearch_key_t)},
    .vectors = {rows, 1024 * sizeof(float), 1024, usearch_scalar_f32_k}});
```

An arena lives where the executor that created it computes, so on a VRAM executor every output an arena holds is __device memory__.
Feed it straight into your next kernel, or bring it back yourself, into the layout you want.
Here the packed rows `get` returns land in a pinned host table with a wider row pitch, in one strided copy:

```c
usearch_rows_view_t rows_on_device = {0};
size_t rows_count = 0;
usearch_arena_t arena = NULL;
usearch_arena_init(&(usearch_arena_init_t){.arena = &arena, .executor = gpu, .outcome = &outcome});
usearch_dense_get(&(usearch_dense_get_t){
    .index = index, .executor = gpu, .outcome = &outcome, .key = row_keys[0],
    .vectors = &rows_on_device, .written = &rows_count, .arena = arena,
    .outputs_scalar = usearch_scalar_f32_k});

size_t const table_pitch = 1280 * sizeof(float);
float* table = NULL;
cudaHostAlloc((void**)&table, rows_count * table_pitch, cudaHostAllocDefault);
cudaMemcpy2DAsync(table, table_pitch, rows_on_device.data, rows_on_device.stride, rows_on_device.stride, rows_count,
                  cudaMemcpyDeviceToHost, stream);
cudaStreamSynchronize(stream);
usearch_arena_free(arena);
```

A search's matches land in its arena too, so on a VRAM executor they are device memory, and a VRAM search reports `usearch_status_device_k` until the devices merge their shards on their own.
A device that runs out of memory for an arena reports `usearch_status_out_of_memory_k` rather than falling back to the host, so where an output lands never depends on how full a device is.

## Serialization

Checkpoints ride the SafeGraphs interchange format, and the path's extension picks the layout: a `.safegraphs` path is one monolith of every shard, a `.safegraphs.json` path is a manifest beside one monolith per shard.
Every lifecycle verb names its action and its residency: `init`, `load` or `view`, on RAM, VRAM or NVMe.

```c
usearch_bytes_view_t path = {(uint8_t const*)"index.safegraphs", 16};
usearch_dense_save(&(usearch_dense_save_t){.index = index, .executor = executor, .outcome = &outcome, .path = path});

usearch_dense_index_t copy = NULL, mapped = NULL;
usearch_dense_load_ram(&(usearch_dense_load_ram_t){.index = &copy, .executor = executor, .outcome = &outcome, .path = path});
usearch_dense_view_ram(&(usearch_dense_view_ram_t){.index = &mapped, .outcome = &outcome, .path = path});
```

A loaded index is a mutable copy, while a viewed one serves the file in place through a memory mapping and refuses every mutating verb with `usearch_status_read_only_k`.
A checkpoint's geometry and metric can be read without loading it:

```c
usearch_dense_metadata_t metadata = {0};
usearch_dense_metadata_read(&(usearch_dense_metadata_read_t){
    .executor = executor, .outcome = &outcome, .path = path, .metadata = &metadata});
```

An index can also be saved into memory as one monolith of every shard, which loads into any shard count.
The bytes land in an arena, so the call needs no size estimate upfront:

```c
usearch_bytes_view_t monolith = {0};
usearch_dense_save_buffer(&(usearch_dense_save_buffer_t){
    .index = index, .executor = executor, .outcome = &outcome, .monolith = &monolith, .arena = arena});
usearch_dense_load_ram_buffer(&(usearch_dense_load_ram_buffer_t){
    .index = &copy, .executor = executor, .outcome = &outcome, .buffer = monolith});
```

## Metrics

USearch comes pre-packaged with NumKong, bringing SIMD-accelerated distance kernels for x86, Arm, RISC-V and GPUs.
The dense metric kinds are:

- `usearch_dense_metric_inner_k` - Inner Product distance, $1 - \sum_i a_i b_i$, unnormalized and scale-sensitive.
- `usearch_dense_metric_angular_k` - Cosine distance, $1 - \frac{\sum_i a_i b_i}{\sqrt{\sum_i a_i^2} \sqrt{\sum_i b_i^2}}$, in $[0, 2]$ and blind to magnitude.
- `usearch_dense_metric_euclidean_k` - Euclidean distance, $\sqrt{\sum_i (a_i - b_i)^2}$, zero only for identical rows.
- `usearch_dense_metric_hamming_k` - Hamming distance, $\sum_i a_i \oplus b_i$, over `usearch_scalar_u1x8_k` bit sets.
- `usearch_dense_metric_jaccard_k` - Jaccard distance, $\frac{\sum_i a_i \oplus b_i}{\sum_i a_i \lor b_i}$, in $[0, 1]$, over `usearch_scalar_u1x8_k` bit sets.

Vectors are quantized on insertion into the index's `quantization`, from `f64` down to `i8`, `u8`, the four 8- and 6-bit floats `e5m2`, `e4m3`, `e3m2` and `e2m3`, and packed 4- and 1-bit integers.

A metric of your own, over payloads of your own, belongs to the C++ baseline tier, whose index takes any batched metric you write; see [`include/usearch/basic/README.md`](../include/usearch/basic/README.md).

## Modalities

`usearch/usearch.h` declares four index families, each in its own header, with the same verbs, executors, outcomes and arenas:

- __Dense__, `dense_index.h` - fixed-width vectors under the metrics above, on RAM and VRAM.
- __Sequences__, `sequence_index.h` - byte strings, UTF-8 text, DNA, RNA and proteins, under Levenshtein, Needleman-Wunsch, Smith-Waterman and their Gotoh affine-gap forms.
- __Matrices__, `matrix_index.h` - sets of equidimensional rows whose count varies per entry, like multi-vector documents or molecules, under Chamfer, Hausdorff, Gromov-Wasserstein, Kabsch and Umeyama.
- __Geospatial__, `geospatial_index.h` - points, paths and rings on and above the ellipsoid, in space and time, searched by approach, overlap, shape, coverage and route.

Variable-length entries travel as Apache Arrow tapes: the elements end to end, and `count + 1` offsets around them.
Every tape names its element kind in `scalar`: `u8` for sequences, `f64` degrees for geospatial points, and any kind for matrix rows.
Sequences, for instance, are bytes with byte offsets:

```c
usearch_sequence_index_t sequences = NULL;
usearch_sequence_init_ram(&(usearch_sequence_init_ram_t){
    .index = &sequences, .outcome = &outcome, .metric = usearch_sequence_metric_levenshtein_utf8_k});

char const tape[] = "kittensittingmitten";
size_t const offsets[4] = {0, 6, 13, 19};
usearch_key_t words[3] = {{{1}}, {{2}}, {{3}}};
usearch_sequence_insert(&(usearch_sequence_insert_t){
    .index = sequences, .executor = executor, .outcome = &outcome,
    .count = 3, .keys = {words, sizeof(usearch_key_t)}, .sequences = {tape, offsets, usearch_scalar_u8_k}});

size_t const query_offsets[2] = {0, 6};
usearch_sequence_search(&(usearch_sequence_search_t){
    .index = sequences, .executor = executor, .outcome = &outcome,
    .count = 1, .queries = {"sitten", query_offsets, usearch_scalar_u8_k},
    .matches = &matches, .arena = arena, .wanted = 3});
```

Matrices share one `columns` width index-wide, so their tape is rows end to end with row offsets.
Geospatial queries are converted into the index's geometry once and reused, which is what `usearch_geospatial_emplace` is for, and `usearch_geospatial_within` reports every entry inside a radius into an arena, however many there turn out to be:

```c
usearch_geospatial_emplaced_t near_station = NULL;
double const station[2] = {51.5308, -0.1238}; // latitude and longitude, in degrees
usearch_geospatial_emplace(&(usearch_geospatial_emplace_t){
    .emplaced = &near_station, .index = places, .executor = executor, .outcome = &outcome,
    .points = station, .point_count = 1});

usearch_matches_tape_view_t nearby = {0}; // `nearby.offsets[1]` entries, their distances in metres
usearch_geospatial_within(&(usearch_geospatial_within_t){
    .index = places, .executor = executor, .outcome = &outcome, .count = 1, .queries = &near_station,
    .mode = usearch_geospatial_mode_approach_k, .axes = usearch_geospatial_axes_horizontal_k,
    .tolerance = {.metres = 500}, .matches = &nearby, .arena = arena});
usearch_geospatial_emplaced_free(near_station);
```

## Filtering with Predicates

Sometimes you may want to cross-reference search results against some external database or filter them by some criteria.
In most engines, you'd have to manually perform paging requests, successively filtering the results.
In USearch you can pass a predicate to the search, which is applied during graph traversal.

```c
usearch_admission_t admits_tenant(usearch_key_t key, void* state) {
    return key.bytes[0] == *(uint8_t const*)state ? usearch_admit_k : usearch_reject_k;
}

uint8_t tenant = 7;
usearch_dense_search(&(usearch_dense_search_t){
    .index = index, .executor = executor, .outcome = &outcome,
    .count = 1, .queries = {query, 0, 128, usearch_scalar_f32_k}, .matches = &matches, .arena = arena, .wanted = 10,
    .predicate = admits_tenant, .predicate_state = &tenant});
```

The predicate runs concurrently on the executor's threads, so it must tolerate that.

## Streaming Search

When a filter is selective enough that no `wanted` can be guessed upfront, a search stream keeps pulling matches in distance order:

```c
usearch_dense_search_stream_t stream = NULL;
usearch_dense_search_stream_init(&(usearch_dense_search_stream_init_t){
    .stream = &stream, .index = index, .outcome = &outcome, .query = {query, 0, 128, usearch_scalar_f32_k}});
do {
    usearch_dense_search_stream_next(&(usearch_dense_search_stream_next_t){
        .index = index, .executor = executor, .outcome = &outcome, .stream = stream,
        .matches = &matches, .arena = arena, .wanted = 10});
    // ... keep the matches that pass, and stop once enough did ...
} while (outcome.status == usearch_status_success_k && matches.counts[0] != 0);
usearch_dense_search_stream_free(stream);
```

A stream keeps the walk's state between pulls, never a thread, so any number of them may stay open on one executor.
It keeps no hold on the index either: every pull passes the index the stream was opened on, and any other is refused.

## Inspecting and Scanning

One call reports an index's frozen geometry and its current counters:

```c
usearch_dense_index_description_t description = {0};
usearch_dense_describe(&(usearch_dense_describe_t){
    .index = index, .executor = executor, .outcome = &outcome, .description = &description});
printf("%zu of %zu entries, %zu bytes, %zu dimensions, kernels: %s\n", description.size, description.capacity,
       description.memory_usage, description.metadata.dimensions, description.capabilities);
```

A key may hold several vectors, up to the index's `values_per_key`, as when a long document is split into chunks.
`usearch_dense_count` reports how many a key holds, and a cursor walks the whole index in storage order:

```c
usearch_dense_cursor_t cursor = NULL;
usearch_dense_cursor_init(&(usearch_dense_cursor_init_t){.cursor = &cursor, .index = index, .outcome = &outcome});
usearch_keys_view_t batch = {0};
for (size_t written = 1; written != 0 && outcome.status == usearch_status_success_k;)
    usearch_dense_cursor_next(&(usearch_dense_cursor_next_t){
        .index = index, .executor = executor, .outcome = &outcome, .cursor = cursor,
        .keys = &batch, .written = &written, .arena = arena, .wanted = 1024});
usearch_dense_cursor_free(cursor);
```

## Performance Tuning

The graph's `connectivity` is frozen when the index is created, while the `expansion_factor`, the breadth of the candidate pool a walk explores, is chosen per call.
Higher expansion values buy recall at the cost of latency; zero derives a default from the connectivity.

```c
usearch_dense_search(&(usearch_dense_search_t){
    .index = index, .executor = executor, .outcome = &outcome,
    .count = 1, .queries = {query, 0, 128, usearch_scalar_f32_k}, .matches = &matches, .arena = arena, .wanted = 10,
    .expansion_factor = 128});
```

`usearch_capabilities_compiled` and `usearch_capabilities_available` name the SIMD and GPU capabilities the library was built with and the machine offers, and `description.capabilities` names the kernels an index dispatched to.

## Design Principles

These are the rules the interface is built toward, and where a verb still falls short of one, the verb is what changes.

- __One shape for every call.__
  Each verb takes one argument struct named after it, so `usearch_dense_search` takes a `usearch_dense_search_t`.
  Fields are set by designator, and every field left out keeps its default, so a call names only what it changes.
- __Synchronous and asynchronous calls look the same.__
  Every call reports through its `outcome`, and names its `completion`: blocking by default, or `usearch_completion_deferred_k` to return once submitted.
  Both are the same struct, so no verb has a second, asynchronous twin, and a thread pool, a GPU stream or a remote server can serve the same verbs without changing a call site.
- __Executors run, indexes store.__
  An executor owns the threads or streams that run calls and the scratch they walk with, while an index owns only its data.
  Calls on one executor start in submission order, and one executor can drive any number of indexes.
- __Memory follows residency.__
  An index keeps its payload where its lifecycle verb put it, an executor keeps its scratch next to its threads or devices, and an arena lives where the executor that created it computes.
  So outputs come back in RAM from a RAM executor and in VRAM from a VRAM executor, and moving them elsewhere is the caller's own copy.
  Callers lend memory, never a way to allocate it: no allocator callback crosses the interface, because large systems hold too many kinds of memory for one ABI to describe them all.
- __Invalid calls should be hard to write.__
  We aim for no `NULL` that switches behaviour, no bitmask that mixes unrelated options, and no field that repeats what another already says.
  Lifecycle verbs name their action and their residency, `init_ram`, `load_vram` or `view_nvme`, instead of taking a tag a residency might not support.
- __Inputs and outputs are views.__
  A zero input stride repeats the first element, as in BLAS, while every output whose length scales with the call is a view the call writes into its arena, so no caller sizes an output or guesses a bound.
  Arenas keep each binding at one verb per operation, never a twin per output style.
- __Failures are statuses.__
  The library never logs or aborts: every failure is a status with a static message, and a verb a residency lacks answers `usearch_status_unsupported_k` rather than missing from the build.
