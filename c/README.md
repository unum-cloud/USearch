# USearch for C

This is the C 99 interface every other binding wraps: the headers in `c/usearch/`, compiled into `libusearch_shared` and `libusearch_static`.
Building the library takes CMake 3.28 and a C++20 compiler, while calling it takes any C99 compiler.
It exposes all four modalities: the engine implements dense indexes, while sequence, shape and geospatial indexes are answered today by a brute-force stand-in only.

## Installation

```sh
cmake -B build_release -D CMAKE_BUILD_TYPE=Release -D USEARCH_BUILD_LIB_C=1 -D USEARCH_DENSE_DTYPES=11 && cmake --build build_release --config Release
```

That leaves both libraries in `build_release/`, carrying all eleven dense storage types; precompiled ones ship on the [releases page](https://github.com/unum-cloud/USearch/releases).
Put `c/` on the include path, or `include/` once installed, where the C headers sit beside the C++ ones, and link either library.
One header, `usearch/usearch.h`, pulls in every modality: `dense_index.h`, `sequence_index.h`, `shapes_index.h` and `geospatial_index.h`.
`ctest --test-dir build_release -R usearch_test_c` runs the smoke test in `c/test.c`.

`-D USEARCH_BUILD_LIB_C_STUB=1` adds a brute-force stand-in under the same library names in `build_release/stub/`.
It answers every verb of every modality by exhaustive scan with the C++ standard library alone, so a binding links and runs its suite in seconds without building the engine.
Its search is exact, its traversal statistics are invented, and its checkpoints are its own memory image, readable by nothing else.

## Quickstart

Every verb takes one argument struct, initialized by designator, so every field left out keeps its default.
Every struct opens in the same order: the handle, `executor`, `outcome`, then `completion`; inputs follow, then outputs and the `arena` they land in, then knobs, and last `progress` and `stats`.
Structs carry no size or version field, so they grow source-compatibly only, and the major version changes whenever one does.

```c
#include <stdio.h>           // For printing.
#include <usearch/usearch.h> // For the win 🚀

int main(void) {
    usearch_outcome_t outcome = {0};
    usearch_executor_t executor = NULL;
    usearch_arena_t arena = NULL;
    usearch_dense_index_t index = NULL;
    usearch_key_t keys[2] = {{.bytes[15] = 1}, {.bytes[15] = 2}};
    float vectors[2][128] = {{0}};
    usearch_matches_view_t matches = {0};

    // Threads live in the executor, the data lives in the index, and outputs land in the arena.
    usearch_executor_for_ram(&(usearch_executor_for_ram_t) {.executor = &executor, .outcome = &outcome});
    if (outcome.status != usearch_status_success_k) goto cleanup;
    usearch_arena_init(&(usearch_arena_init_t) {.arena = &arena, .executor = executor, .outcome = &outcome});
    if (outcome.status != usearch_status_success_k) goto cleanup;
    usearch_dense_init_ram(&(usearch_dense_init_ram_t) {.index = &index,
                                                        .outcome = &outcome,
                                                        .metric = usearch_dense_metric_angular_k,
                                                        .quantization = usearch_scalar_bf16_k,
                                                        .dimensions = 128});
    if (outcome.status != usearch_status_success_k) goto cleanup;

    // Populate:
    usearch_dense_insert(
        &(usearch_dense_insert_t) {.index = index,
                                   .executor = executor,
                                   .outcome = &outcome,
                                   .count = 2,
                                   .keys = {keys, sizeof(usearch_key_t)},
                                   .vectors = {vectors, sizeof(vectors[0]), 128, usearch_scalar_f32_k}});
    if (outcome.status != usearch_status_success_k) goto cleanup;

    // Search one query; a zero stride would repeat it across a larger `count`.
    usearch_dense_search(&(usearch_dense_search_t) {.index = index,
                                                    .executor = executor,
                                                    .outcome = &outcome,
                                                    .count = 1,
                                                    .queries = {vectors[0], 0, 128, usearch_scalar_f32_k},
                                                    .matches = &matches,
                                                    .arena = arena,
                                                    .wanted = 10});
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

Every call here blocks, so its outcome is final by the time it returns.

## Completion & Deferred Calls

Every call reports through its `outcome` and names its `completion` in the same struct, so no verb has a second, asynchronous twin.
`usearch_completion_blocking_k`, the default, returns with the final status stored and every output readable.
`usearch_completion_deferred_k` returns once submitted, as an io_uring entry does, and the call finishes on its executor.
A thread pool, a GPU stream or a remote server can serve the same verbs without changing a call site.

Submission copies the argument struct, but everything it points to stays borrowed until the outcome is final.
So many calls can be deferred at once, each with its own outcome and its own arena, which it holds until then:

```c
usearch_matches_view_t matches[4];
usearch_outcome_t outcomes[4] = {{0}};
for (size_t shard = 0; shard != 4; ++shard)
    usearch_dense_search(&(usearch_dense_search_t) {
        .index = shards[shard],
        .executor = executor,
        .outcome = &outcomes[shard],
        .completion = usearch_completion_deferred_k,
        .count = 32,
        .queries = {queries, 128 * sizeof(float), 128, usearch_scalar_f32_k},
        .matches = &matches[shard],
        .arena = arenas[shard],
        .wanted = 10});
for (size_t shard = 0; shard != 4; ++shard) usearch_outcome_wait(&outcomes[shard]);
```

The status reads `usearch_status_deferred_k` until the call is final, even for a call refused at submission.
Then the library stores the message and the final status with release ordering, and touches nothing the struct names after that.
The caller reads it with acquire ordering: `usearch_outcome_status` polls and `usearch_outcome_wait` blocks on an atomic wait, both returning the status.
Those two are the only safe readers of a deferred outcome, since the status word is atomic only by their contract.
Never wait on a call's own outcome from inside its progress handler or predicate, since the call finishes only after they return.
Executors own their streams, so a caller orders dependent calls by waiting on the first.

Each executor tracks its deferred calls in a fixed table of `deferred_limit` slots, set when it is created, zero taking a default.
A deferred submission that finds the table full finishes at once with `usearch_status_contended_k`, so submitting never allocates, and blocking calls take no slot at all.
Every `*_free` waits until each deferred call naming its handle is final, so a handle may be dropped at any moment.

## Executors

An executor owns the threads or device streams that run calls, the traversal scratch they walk with, and the pinned bounce buffer staging passes through.
One executor serves any number of indexes, of any quantization and metric, since its scratch is sized from geometry, grown to the widest asked of it, and kept between calls.
Calls on one executor start in submission order, while a handle admits any number of concurrent callers: the executor orders, never the index.

```c
int32_t const sockets[2] = {0, 1};
usearch_executor_for_ram(&(usearch_executor_for_ram_t) {.executor = &executor,
                                                        .outcome = &outcome,
                                                        .ram = {sockets, 2}, // NUMA nodes whose cores it spawns on
                                                        .threads = 0});      // zero takes 7/8 of those cores
```

Threads interleave across the named domains, so a half-size executor takes half of every socket rather than all of one, and the default stops at 7/8 because full occupancy starves ordered publication.
`usearch_executor_for_vram` opens one stream per device, or borrows the caller's, and `usearch_executor_for_nvme` spawns one thread per submission ring.
An application with its own ForkUnion pool lends it with `usearch_executor_on_pool` instead, though the engine refuses borrowed pools with `usearch_status_unsupported_k` for now.
An executor's kind must match the residency of every handle it drives, and a mismatch reports `usearch_status_device_k`.

A `NULL` executor is accepted by every verb and runs the call on the caller's own thread, final when it returns whatever its `completion`, which suits size lookups and other short calls.
A blocking call on an executor runs after every earlier call finishes and before any later one starts, so callers wanting real parallelism give each thread its own executor.
`usearch_executor_drain` waits for everything submitted before it, then sets the executor's idle mood: `usearch_mood_parked_k` lets idle threads sleep, `usearch_mood_spinning_k` keeps them hot.
A blocking verb or a drain called from inside a progress handler or predicate of a call on the same executor reports `usearch_status_contended_k` rather than waiting on itself.
`usearch_executor_free` drains every call before it releases the threads, and arenas the executor made stay valid until freed.

## Arenas

Every output whose length scales with the call lands in an arena: the matches of a search, every vector stored under one key, every pair a sparse product scored, a monolith, a cursor's batch.
An arena is memory the library allocates and the caller passes to call after call, each of which rewinds it and writes views into it, growing only when the new outputs outsize everything it held.
An executor makes it where it computes, and a `NULL` executor makes host pages for calls on the calling thread; it may outlive the executor that made it.
Nothing is allocated until a call writes into it, and it keeps its high-water mark, so a steady loop allocates once.
Only fixed-size reports, such as stats, descriptions and `written` counts, stay fields the caller points at.
Arenas keep the interface at one verb per operation, never a twin per output style.

```c
usearch_arena_t arena = NULL;
usearch_arena_init(&(usearch_arena_init_t) {.arena = &arena, .executor = executor, .outcome = &outcome});
for (size_t i = 0; i != 2; ++i) {
    usearch_rows_view_t rows = {0};
    size_t rows_count = 0;
    usearch_dense_get(&(usearch_dense_get_t) {.index = index,
                                              .executor = executor,
                                              .outcome = &outcome,
                                              .key = keys[i],
                                              .vectors = &rows,
                                              .written = &rows_count,
                                              .arena = arena,
                                              .outputs_scalar = usearch_scalar_f32_k});
    if (outcome.status != usearch_status_success_k) break;
    // `rows.data` holds `rows_count` vectors `rows.stride` bytes apart,
    // valid until the next call given `arena`
}
usearch_arena_free(arena);
```

Any verb may reuse any arena of its residency, so a search and a get can share one, as long as the first call's views are not read after the second starts.
To keep one call's outputs alive past the next, pass the next call another arena.
An arena serves one call at a time: a blocking call borrows it, a deferred call holds it until final, and a second call naming it meanwhile finishes `usearch_status_contended_k`.
A `NULL` arena, or inputs pointing into the arena a call is about to rewind, refuse with `usearch_status_invalid_argument_k`, and a call that fails leaves its outputs `NULL`.

## Keys

A `usearch_key_t` is sixteen opaque bytes, enough for a UUID, a content hash, or a 64-bit integer with room to spare, and spelled canonically as a lowercase dashed UUID.
Integer identities sit big-endian in the last bytes with the rest zeroed, which is how the other bindings convert integers:

```c
usearch_key_t key_of(uint64_t value) {
    usearch_key_t key = {{0}};
    for (unsigned byte = 0; byte != 8; ++byte) key.bytes[15 - byte] = (uint8_t)(value >> (byte * 8));
    return key;
}
```

Keys need not be unique: an index admits up to its `values_per_key` entries under one key, one by default, and the count verbs report how many a key holds.
A batch passes its keys as a `usearch_keys_view_t`, a pointer and a byte stride, where a zero stride repeats the first key for every row.

## Errors

The library never logs or aborts: every failure is a status in the call's `usearch_outcome_t`, with a static message carrying the detail the status cannot.
Messages are never freed, and are `NULL` exactly on success.
`usearch_status_name` spells a status as its lowercase dashed name, like `invalid-argument`, and `usearch_status_from_name` reads one back.
The statuses worth branching on:

- `usearch_status_unsupported_k`: the verb is declared but this residency or build does not implement it, so no retry can help.
- `usearch_status_read_only_k`: a mutating verb on a viewed handle.
- `usearch_status_contended_k`: refused rather than waited for, like a full deferred table or an arena serving another call.
- `usearch_status_cancelled_k`: a progress handler stopped the call, which published nothing.
- `usearch_status_device_k`: an executor, arena or handle on another residency or device than the call needs.
- `usearch_status_invalid_argument_k`: a required pointer left `NULL`, inputs inside the arena the call rewinds, a name no table spells.

The set describes the project's failures, not only this library's exits, so a binding validating before the call reports what the library would have.
We aim for no `NULL` that switches behaviour, no bitmask that mixes unrelated options, and no field that repeats what another already says.

## Residency

An index lives in RAM, VRAM or on NVMe, and every lifecycle verb names its action and its residency instead of taking a tag a residency might not support:

- `init` starts empty and mutable: `usearch_dense_init_ram`, `usearch_dense_init_vram`.
- `load` copies a checkpoint in: `usearch_dense_load_ram`, `usearch_dense_load_vram`, and their `_buffer` forms.
- `view` serves one in place, read-only: `usearch_dense_view_ram`, `usearch_dense_view_nvme`.

The options name where an index lives, never who walks it.
`usearch_ram_options_t` names NUMA nodes, one shard per node, so a dual-socket box gets two graphs whose walks stay local; payload blocks claim huge pages where a pool is provisioned.
`usearch_vram_options_t` names device ordinals, one shard per device and never a replica, so an index outgrowing them fails rather than paging to the host.
`usearch_nvme_options_t` names paths: a filesystem path selects a kernel I/O engine, a transport identifier like `trtype:PCIe traddr:0000:5e:00.0` selects SPDK in a `-D USEARCH_BUILD_SPDK=1` build, and a FUSE mount over a bucket reads as an ordinary path.

Memory follows residency: an index keeps its payload where its lifecycle verb put it, an executor keeps its scratch next to its threads or devices, and an arena lives where the executor that created it computes.
So outputs come back in RAM from a RAM executor and in VRAM from a VRAM executor, and moving them elsewhere is the caller's own copy.
Callers lend memory, never a way to allocate it: no allocator callback crosses the interface, because large systems hold too many kinds of memory for one ABI to describe them all.

Every dense verb is exported for every residency, and the engine answers what a residency lacks with `usearch_status_unsupported_k`:

- RAM answers every verb but `usearch_dense_view_ram`, `usearch_dense_search_stream_init` and `usearch_dense_merge`.
- VRAM answers insertion, refinement, counting, `multiply`, `maxsim`, attention and the serialization family, while `get`, `save_buffer`, cursors and exact scans answer `usearch_status_unsupported_k`, and a search reports `usearch_status_device_k`, as [GPUs](#gpus) explains.
- NVMe answers every read verb as RAM does, refuses every mutating one with `usearch_status_read_only_k`, and refuses saving, since a view already is its checkpoint.

### GPUs

A VRAM index lives on the devices its `vram` options name, one shard per device, and a VRAM executor drives it through one stream per device.
Passing your own streams lets USearch work join the rest of your pipeline, though the engine refuses borrowed streams with `usearch_status_unsupported_k` for now:

```c
#include <cuda_runtime.h>

cudaStream_t stream;
cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
int32_t const device = 0;
usearch_executor_t gpu = NULL;
usearch_executor_for_vram(&(usearch_executor_for_vram_t) {
    .executor = &gpu, .outcome = &outcome, .vram = {&device, 1}, .streams = (void* const*)&stream});

usearch_dense_index_t index = NULL;
usearch_dense_init_vram(&(usearch_dense_init_vram_t) {.index = &index,
                                                      .outcome = &outcome,
                                                      .metric = usearch_dense_metric_inner_k,
                                                      .quantization = usearch_scalar_bf16_k,
                                                      .dimensions = 1024,
                                                      .vram = {&device, 1}});
usearch_dense_reserve(&(usearch_dense_reserve_t) {
    .index = index, .executor = gpu, .outcome = &outcome, .capacity = 1000000});
```

The library classifies every pointer it is given through the driver, so inputs may live anywhere:
rows already on the device are read in place, pinned host rows upload directly, and only pageable host rows pass through the executor's pinned bounce buffer.
Pin the rows you upload often, and the bounce disappears:

```c
float* rows = NULL;
cudaHostAlloc((void**)&rows, 4096 * 1024 * sizeof(float), cudaHostAllocDefault);
// ... fill 4096 rows ...
usearch_dense_insert(&(usearch_dense_insert_t) {.index = index,
                                                .executor = gpu,
                                                .outcome = &outcome,
                                                .count = 4096,
                                                .keys = {row_keys, sizeof(usearch_key_t)},
                                                .vectors = {rows, 1024 * sizeof(float), 1024, usearch_scalar_f32_k}});
```

An arena lives where the executor that created it computes, so on a VRAM executor every output an arena holds is __device memory__.
Feed it straight into your next kernel, or bring it back yourself, into the layout you want.
Here the packed rows `get` returns land in a pinned host table with a wider row pitch, in one strided copy, once VRAM `get` lands in device arenas:

```c
usearch_rows_view_t rows_on_device = {0};
size_t rows_count = 0;
usearch_arena_t arena = NULL;
usearch_arena_init(&(usearch_arena_init_t) {.arena = &arena, .executor = gpu, .outcome = &outcome});
usearch_dense_get(&(usearch_dense_get_t) {.index = index,
                                          .executor = gpu,
                                          .outcome = &outcome,
                                          .key = row_keys[0],
                                          .vectors = &rows_on_device,
                                          .written = &rows_count,
                                          .arena = arena,
                                          .outputs_scalar = usearch_scalar_f32_k});

size_t const table_pitch = 1280 * sizeof(float);
float* table = NULL;
cudaHostAlloc((void**)&table, rows_count* table_pitch, cudaHostAllocDefault);
cudaMemcpy2DAsync(table, table_pitch, rows_on_device.data, rows_on_device.stride, rows_on_device.stride, rows_count,
                  cudaMemcpyDeviceToHost, stream);
cudaStreamSynchronize(stream);
usearch_arena_free(arena);
```

A search's matches land in its arena too, so on a VRAM executor they are device memory, and a VRAM search reports `usearch_status_device_k` until the devices merge their shards on their own.
A device that runs out of memory for an arena reports `usearch_status_out_of_memory_k` rather than falling back to the host, so where an output lands never depends on how full a device is.
Free a device arena once your own device work on its outputs has finished.

## Serialization

Checkpoints ride the SafeGraphs interchange format, and the path's extension picks the layout: a `.safegraphs` path is one monolith of every shard, a `.safegraphs.json` path is a manifest beside one monolith per shard.
Loading reads the content, never the extension, so either form loads into any shard count, part `p` landing on shard `p` modulo the shard count.

```c
usearch_bytes_view_t path = {(uint8_t const*)"index.safegraphs", 16};
usearch_dense_save(&(usearch_dense_save_t) {.index = index, .executor = executor, .outcome = &outcome, .path = path});

usearch_dense_index_t copy = NULL, mapped = NULL;
usearch_dense_load_ram(&(usearch_dense_load_ram_t) {
    .index = &copy, .executor = executor, .outcome = &outcome, .path = path});
usearch_dense_view_ram(&(usearch_dense_view_ram_t) {.index = &mapped, .outcome = &outcome, .path = path});
```

A loaded index is a mutable copy, while a viewed one serves the file in place through a memory mapping and refuses every mutating verb with `usearch_status_read_only_k`.
The engine's RAM shards cannot serve a mapping yet, so it answers `usearch_dense_view_ram` with `usearch_status_unsupported_k`, while `usearch_dense_view_nvme` serves checkpoints in place today.
A cancelled or failed save leaves a file without its header or parts without a manifest, which every reader rejects and the caller removes, and a cancelled or failed load creates no handle.
A checkpoint's geometry and metric can be read without loading it:

```c
usearch_dense_metadata_t metadata = {0};
usearch_dense_metadata_read(&(usearch_dense_metadata_read_t) {
    .executor = executor, .outcome = &outcome, .path = path, .metadata = &metadata});
```

An index can also be saved into memory as one monolith of every shard, which loads into any shard count.
The bytes land in an arena, so the call needs no size estimate upfront:

```c
usearch_bytes_view_t monolith = {0};
usearch_dense_save_buffer(&(usearch_dense_save_buffer_t) {
    .index = index, .executor = executor, .outcome = &outcome, .monolith = &monolith, .arena = arena});
usearch_dense_load_ram_buffer(&(usearch_dense_load_ram_buffer_t) {
    .index = &copy, .executor = executor, .outcome = &outcome, .buffer = monolith});
```

## Dense Indexes

A dense index holds fixed-width vectors under one metric, with the `dimensions`, `quantization`, `connectivity` and `values_per_key` its constructor froze.

### Rows & Scalars

A batch of rows is a `usearch_rows_view_t`: a base pointer, a byte stride, the values per row, and the scalar kind they hold, converted on the way in where it differs from storage.
A zero stride repeats the first row, as in BLAS, and the call's `count` alone is the batch axis.
Vectors are quantized on insertion into the index's `quantization`, from `f64` down to `i8`, `u8`, the four 8- and 6-bit floats `e5m2`, `e4m3`, `e3m2` and `e2m3`, and packed 4- and 1-bit integers.
The library compiles the first `USEARCH_DENSE_DTYPES` storage types of `bf16`, `u1x8`, `i8`, `e4m3`, `f16`, `u8`, `e5m2`, `i4x2`, `u4x2`, `e3m2` and `e2m3`, and refuses the rest with `usearch_status_unsupported_k`.
The precompiled binaries carry all eleven, while a CMake build takes the first four unless configured with `-D USEARCH_DENSE_DTYPES=11`.

USearch comes pre-packaged with NumKong, bringing SIMD-accelerated distance kernels for x86, Arm, RISC-V and GPUs.
The dense metric kinds are:

- `usearch_dense_metric_inner_k` - Inner Product distance, $1 - \sum_i a_i b_i$, unnormalized and scale-sensitive.
- `usearch_dense_metric_angular_k` - Cosine distance, $1 - \frac{\sum_i a_i b_i}{\sqrt{\sum_i a_i^2} \sqrt{\sum_i b_i^2}}$, in $[0, 2]$ and blind to magnitude.
- `usearch_dense_metric_euclidean_k` - Euclidean distance, $\sqrt{\sum_i (a_i - b_i)^2}$, zero only for identical rows.
- `usearch_dense_metric_hamming_k` - Hamming distance, $\sum_i a_i \oplus b_i$, over `usearch_scalar_u1x8_k` bit sets.
- `usearch_dense_metric_jaccard_k` - Jaccard distance, $\frac{\sum_i a_i \oplus b_i}{\sum_i a_i \lor b_i}$, in $[0, 1]$, over `usearch_scalar_u1x8_k` bit sets.

`usearch_scalar_name`, `usearch_dense_metric_name` and their `_from_name` inverses spell these sets the way every binding parses them.

### Search

A search writes a `usearch_matches_view_t` into its arena: `count` rows of `columns` packed columns, row `i` holding `counts[i]` matches from `keys[i * columns]` and `scores[i * columns]` on, nearest first.
`wanted` bounds the columns, the "k" in "kANN".
The graph's `connectivity` is frozen when the index is created, while the `expansion_factor`, the breadth of the candidate pool a walk explores, is chosen per call.
Higher expansion values buy recall at the cost of latency; zero derives a default from the connectivity.

```c
usearch_dense_traversal_stats_t stats = {0};
usearch_dense_search(&(usearch_dense_search_t) {.index = index,
                                                .executor = executor,
                                                .outcome = &outcome,
                                                .count = 1,
                                                .queries = {query, 0, 128, usearch_scalar_f32_k},
                                                .matches = &matches,
                                                .arena = arena,
                                                .wanted = 10,
                                                .expansion_factor = 128,
                                                .stats = &stats});
```

`.traversal = usearch_traversal_scan_k` scores every published row instead of walking, the exact ground truth a walk approximates, on RAM and NVMe.
Every verb that walks the graph fills the optional `stats` with the members it visited, the distances it computed, and its restarts, a connectivity health signal rather than a cost.

### Filtering & Progress

Sometimes you may want to cross-reference search results against some external database or filter them by some criteria.
In most engines, you'd have to manually perform paging requests, successively filtering the results.
In USearch you can pass a predicate to the search, which is applied during graph traversal.

```c
void admits_tenant(usearch_key_t const* keys, usearch_admission_t* admissions, size_t count, void* state) {
    for (size_t i = 0; i != count; ++i)
        admissions[i] = keys[i].bytes[0] == *(uint8_t const*)state ? usearch_admit_k : usearch_reject_k;
}

uint8_t tenant = 7;
usearch_dense_search(&(usearch_dense_search_t) {.index = index,
                                                .executor = executor,
                                                .outcome = &outcome,
                                                .count = 1,
                                                .queries = {query, 0, 128, usearch_scalar_f32_k},
                                                .matches = &matches,
                                                .arena = arena,
                                                .wanted = 10,
                                                .predicate = {admits_tenant, &tenant}});
```

The predicate judges a batch of candidates per call, a quad of them at a time during a walk, so a handler backed by an external store answers each batch in one round trip.

Rejected entries still route the walk, so recall holds at low selectivity, and the predicate judges every scored candidate, concurrently on the executor's threads, so it must be fast and tolerate that.
An exact scan asks it only about rows closer than the worst match a query already holds, which makes the scan the fallback for filters too selective for a walk to cross.
A host predicate cannot narrow a device walk, so a VRAM search with one reports `usearch_status_device_k`.

Long calls take a `usearch_progress_t`: a handler, which is also the only way to stop one, its state, and how many reports it hears:

```c
usearch_continuation_t stop_when_asked(size_t completed, size_t total, void* state) {
    return *(int const*)state ? usearch_cancel_k : usearch_continue_k;
}

int stop = 0;
usearch_dense_search(&(usearch_dense_search_t) {
    .index = index, .executor = executor, .outcome = &outcome, .count = 1000000, .queries = queries,
    .wanted = 10, .arena = arena, .matches = &matches, .progress = {stop_when_asked, &stop, 20}});
```

Counts are in whatever units the verb consumes, rows for an insertion, queries for a search, shards for a checkpoint, and the handler is never called concurrently or after the call is final.
A call reports at most `steps` times, the last at completion, and zero steps derive to ten.
A batch verb hands its engine one slice per step, and a call without a handler runs as one slice, since every engine bounds its own scratch.
A stopped call reports `usearch_status_cancelled_k` and leaves its outputs `NULL`.
An insertion claims every row before its first report, so its handler only reports, and all its rows publish at once.

### Retrieval: Get, Cursors & Streams

`usearch_dense_get` writes every vector stored under one key into an arena, shard by shard and in insertion order within each, as the [Arenas](#arenas) example shows; an absent key leaves `vectors->data` `NULL`.
A cursor walks the whole index in storage order, its keys and, when asked, its rows landing in an arena batch by batch:

```c
usearch_dense_cursor_t cursor = NULL;
usearch_dense_cursor_init(&(usearch_dense_cursor_init_t) {.cursor = &cursor, .index = index, .outcome = &outcome});
usearch_keys_view_t batch = {0};
for (size_t written = 1; written != 0 && outcome.status == usearch_status_success_k;)
    usearch_dense_cursor_next(&(usearch_dense_cursor_next_t) {.index = index,
                                                              .executor = executor,
                                                              .outcome = &outcome,
                                                              .cursor = cursor,
                                                              .keys = &batch,
                                                              .written = &written,
                                                              .arena = arena,
                                                              .wanted = 1024});
usearch_dense_cursor_free(cursor);
```

Each shard's publication watermark is read as the walk enters it, so a row once passed is never revisited, while a later shard may carry rows published after the walk began.

When a filter is selective enough that no `wanted` can be guessed upfront, a search stream keeps pulling matches in distance order:

```c
usearch_dense_search_stream_t stream = NULL;
usearch_dense_search_stream_init(&(usearch_dense_search_stream_init_t) {
    .stream = &stream, .index = index, .outcome = &outcome, .query = {query, 0, 128, usearch_scalar_f32_k}});
do {
    usearch_dense_search_stream_next(&(usearch_dense_search_stream_next_t) {.index = index,
                                                                            .executor = executor,
                                                                            .outcome = &outcome,
                                                                            .stream = stream,
                                                                            .matches = &matches,
                                                                            .arena = arena,
                                                                            .wanted = 10});
    // ... keep the matches that pass, and stop once enough did ...
} while (outcome.status == usearch_status_success_k && matches.counts[0] != 0);
usearch_dense_search_stream_free(stream);
```

No residency implements streams yet, so `usearch_dense_search_stream_init` answers `usearch_status_unsupported_k` today.
A stream keeps the walk's state between pulls, never a thread, so any number of them may stay open on one executor.
Cursors and streams keep no hold on the index either: every pull passes the index it was opened on, any other is refused with `usearch_status_invalid_argument_k`, and an overlapping pull reports `usearch_status_contended_k`.

### Refinement

`usearch_dense_refine` re-selects the neighbors of every stored entry against the finished graph, improving those chosen while the graph around them was incomplete.
A cheap insertion pass followed by one refinement reaches a better graph in less total time than one expensive pass, so a wide `expansion_factor` belongs here rather than in `usearch_dense_insert`.
Re-selection is idempotent and claims no identity, so a cancelled pass leaves the graph correct and merely less improved, reports no `usearch_status_cancelled_k`, and still re-links any entry no search would reach.
That makes refining under a time budget converge across calls:

```c
size_t changed = 0;
do {
    usearch_dense_refine(&(usearch_dense_refine_t) {.index = index,
                                                    .executor = executor,
                                                    .outcome = &outcome,
                                                    .changed = &changed,
                                                    .expansion_factor = 256});
} while (outcome.status == usearch_status_success_k && changed > 100);
```

Searches may run throughout on a RAM handle, while a VRAM handle rewrites neighbors in place and admits no overlapping search.

### Merging & Splitting

`usearch_dense_merge` absorbs an array of donor handles into one index by landmark stitching, cheaper than re-inserting their rows, at the wall clock of the largest donor.
Donors are consumed and nulled as they go, and cancellation stops between donors, so the array itself says which handles the caller still owns and must free.
No residency absorbs donors yet, so the engine answers `usearch_status_unsupported_k` today, and splitting is not in the C library yet.

### Sparse Multiplication

`usearch_dense_multiply` is an approximate $A \times B^T$ in the shape of BLAS `?GEMM`, with the indexed rows as $B$, reporting every pair the walks scored rather than only the nearest.
The output pattern is whatever the walk touched, so its cost is roughly `count × expansion × connectivity × dimensions`, sublinear in the indexed rows, and the pairs it never reaches are the low and negative dots.
Only the walks learn how many pairs there are, so keys, products and Apache Arrow offsets land in the arena as one ragged output:

```c
usearch_matches_tape_view_t products = {0};
size_t required = 0;
usearch_dense_multiply(&(usearch_dense_multiply_t) {.index = index,
                                                    .executor = executor,
                                                    .outcome = &outcome,
                                                    .count = m,
                                                    .left = {left_rows, 128 * sizeof(float), 128, usearch_scalar_f32_k},
                                                    .products = &products,
                                                    .required = &required,
                                                    .arena = arena,
                                                    .wanted = 256});
for (size_t row = 0; row != m; ++row)
    for (size_t pair = products.offsets[row]; pair != products.offsets[row + 1]; ++pair)
        accumulate(row, products.keys[pair], products.scores[pair]);
```

Pairs arrive in traversal order without duplicates, a non-zero `wanted` keeps each row's first pairs, and `required` reports how many there were before the cut.
The linear-algebra verbs run on `usearch_dense_metric_inner_k` alone, since products and softmaxes need logits rather than distances, and any other kind refuses with `usearch_status_config_mismatch_k`.

### Late Interaction

`usearch_dense_maxsim` is ColBERT scoring, and it is what `values_per_key` above one is for: a document is one key holding its token rows, and its score sums each query row's best match among them.
The graph stays over individual tokens, so no set-to-set metric is evaluated during the walk, documents need not share a row count, and queries are padded to a uniform `rows_per_query`:

```c
usearch_matches_view_t documents = {0};
usearch_dense_maxsim(&(usearch_dense_maxsim_t) {
    .index = index,
    .executor = executor,
    .outcome = &outcome,
    .count = 4,
    .queries = {query_tokens, 128 * sizeof(float), 128, usearch_scalar_f32_k},
    .rows_per_query = 32,
    .matches = &documents,
    .arena = arena,
    .wanted = 10});
```

Each query row contributes only the tokens its walk ranked, so a key can score below its exact value but never above it, and widening the expansion narrows the gap.

### Attention

`usearch_dense_attend` folds Flash Attention's online softmax over the keys each walk scores, with the graph replacing the sweep over key tiles and no top-k cut, so the tail the walk scores brings the result toward exact attention.
One index holds one layer's one key-value head, row `t * head_count + h` is head `h` of token `t`, and the `head_count` query heads of grouped-query attention share one walk per token.
The index stores no values, so `value_lookup` resolves the keys of every scored batch to rows as wide as the keys, in the index's own scalar kind:

```c
void value_rows_of(void* state, usearch_key_t const* keys, size_t count, usearch_scalar_t scalar, size_t columns,
                   void const** rows) {
    for (size_t i = 0; i != count; ++i) rows[i] = find_value_row(state, keys[i]); // a NULL row refuses the call
}

usearch_dense_softmax_state_t carry[64 * 8];
usearch_rows_view_t attended = {0};
usearch_dense_attend(&(usearch_dense_attend_t) {.index = index,
                                                .executor = executor,
                                                .outcome = &outcome,
                                                .count = 64 * 8,
                                                .queries = {query_rows, 128 * 2, 128, usearch_scalar_bf16_k},
                                                .head_count = 8,
                                                .value_lookup = value_rows_of,
                                                .value_lookup_state = &values,
                                                .carry = carry,
                                                .outputs = &attended,
                                                .arena = arena});
```

On RAM `value_lookup` is a host function, and on VRAM a `__device__` one whose address comes from `cudaMemcpyFromSymbol`; passing the wrong kind is undefined, never refused.
`carry` holds each row's running maximum and partition, which merges this branch with the recent window kept out of the index, attention sinks, or a second index of one context:
$m = \max(m_1, m_2)$, $a_i = l_i e^{m_i - m}$, $l = a_1 + a_2$, $o = (a_1 o_1 + a_2 o_2) / l$.
Attention never writes the graph, so it overlaps insertions freely, the publication watermark acting as the causal mask.

### Introspection

One call reports an index's frozen geometry and its current counters:

```c
usearch_dense_index_description_t description = {0};
usearch_dense_describe(&(usearch_dense_describe_t) {
    .index = index, .executor = executor, .outcome = &outcome, .description = &description});
printf("%zu of %zu entries, %zu bytes, %zu dimensions, kernels: %s\n", description.size, description.capacity,
       description.memory_usage, description.metadata.dimensions, description.capabilities);
```

`usearch_dense_count` reports how many entries a key holds, as when a long document is split into chunks.
`usearch_dense_health` audits how much of the graph a search reaches, sweeping forward from as many landmarks as one search draws:

```c
usearch_dense_reachability_t reachability = {0};
usearch_dense_health(&(usearch_dense_health_t) {
    .index = index, .executor = executor, .outcome = &outcome, .reachability = &reachability});
// `reachability.reachable` of `reachability.audited` entries are found by some search
```

`usearch_capabilities_compiled` and `usearch_capabilities_available` name the SIMD and GPU capabilities the library was built with and the machine offers, and `description.capabilities` names the kernels an index dispatched to.
`usearch_version` reports the linked library's version, which may differ from the `USEARCH_VERSION_MAJOR` family of macros the caller compiled against.

## Sequence Indexes

A sequence index searches byte strings, UTF-8 text, DNA, RNA and proteins under edit distances and alignments, in RAM only.
Variable-length entries travel as Apache Arrow tapes, `usearch_tape_view_t`: the elements end to end, and `count + 1` offsets around them, so sequences are bytes with byte offsets:

```c
usearch_sequence_index_t sequences = NULL;
usearch_sequence_init_ram(&(usearch_sequence_init_ram_t) {
    .index = &sequences, .outcome = &outcome, .metric = usearch_sequence_metric_levenshtein_utf8_k});

char const tape[] = "kittensittingmitten";
size_t const offsets[4] = {0, 6, 13, 19};
usearch_key_t words[3] = {{.bytes[15] = 1}, {.bytes[15] = 2}, {.bytes[15] = 3}};
usearch_sequence_insert(&(usearch_sequence_insert_t) {.index = sequences,
                                                      .executor = executor,
                                                      .outcome = &outcome,
                                                      .count = 3,
                                                      .keys = {words, sizeof(usearch_key_t)},
                                                      .sequences = {tape, offsets, usearch_scalar_u8_k}});

size_t const query_offsets[2] = {0, 6};
usearch_sequence_search(&(usearch_sequence_search_t) {.index = sequences,
                                                      .executor = executor,
                                                      .outcome = &outcome,
                                                      .count = 1,
                                                      .queries = {"sitten", query_offsets, usearch_scalar_u8_k},
                                                      .matches = &matches,
                                                      .arena = arena,
                                                      .wanted = 3});
```

The kinds are Levenshtein over bytes or UTF-8 code points, Needleman-Wunsch and Smith-Waterman with linear gaps, their Gotoh forms with affine gaps, and overlap, the share of fixed-width windows one side misses of the other.
The alignment kinds take `usearch_sequence_weights_t`, a 256×256 substitution matrix with gap costs, and the overlap kind takes `window_widths` in bytes, each refused by every other kind with `usearch_status_config_mismatch_k`.
The local Smith-Waterman kinds only rank: each is served from an index built under its global counterpart, which the library picks.
The verbs mirror the dense ones: `init_ram`, `load_ram`, `view_ram`, `reserve`, `clear`, `insert`, `refine`, `search` with a predicate, `get` and `cursor_next` writing tapes into an arena, `merge`, `save`, `metadata_read` and `describe`, while `count` also reports a key's total and largest byte lengths.
A key already holding `values_per_key` entries rejects the whole insertion with `usearch_status_overflow_k`.
The engine does not export these verbs yet, so a program calling them links against the stand-in, which scores every kind by textbook dynamic programming, quadratic and unbanded, and reports alignment similarities negated.

## Shape Indexes

A shape index searches 3D point clouds of any size, like molecules, conformers and protein backbones, in RAM or VRAM.
Its tape is points end to end, `x`, `y` and `z` interleaved, with offsets that count points rather than scalars:

```c
usearch_shapes_index_t conformers = NULL;
usearch_shapes_init_ram(&(usearch_shapes_init_ram_t) {.index = &conformers,
                                                      .outcome = &outcome,
                                                      .metric = usearch_shapes_metric_wasserstein_procrustes_k,
                                                      .quantization = usearch_scalar_f16_k});

float const atoms[5][3] = {{0, 0, 0}, {1.1f, 0, 0}, {0, 0, 0}, {1.5f, 0, 0}, {0, 1.5f, 0}};
size_t const point_offsets[3] = {0, 2, 5}; // a 2-atom and a 3-atom entry
usearch_shapes_insert(&(usearch_shapes_insert_t) {.index = conformers,
                                                  .executor = executor,
                                                  .outcome = &outcome,
                                                  .count = 2,
                                                  .keys = {molecules, sizeof(usearch_key_t)},
                                                  .shapes = {atoms, point_offsets, usearch_scalar_f32_k}});
```

Chamfer and Hausdorff pair every point with its nearest partner after aligning the principal axes under all eight sign flips, keeping the best.
Wasserstein-Procrustes pairs points by an exact transport plan, alternated with the best rotation or reflection.
Gromov-Wasserstein compares each side's internal distances.
All four are invariant to rotation, reflection, translation and point order.
A non-zero `points` freezes the count, rejecting insertions of any other with `usearch_status_shape_mismatch_k`.
The verbs are the sequence ones plus `init_vram` and `load_vram`, where `get` and `cursor_next` convert to an `outputs_scalar` and return each shape in its inserted pose, and `count` reports a key's total and largest point counts.
The engine does not export these verbs yet, so a program calling them links against the stand-in, which scores Chamfer and Hausdorff in the frame the points arrive in, refuses the two transport kinds with `usearch_status_unknown_kind_k`, refuses quantizations other than `f16` and `f32` with `usearch_status_unsupported_k`, and refuses every VRAM verb with `usearch_status_device_k`.

## Geospatial Indexes

A geospatial index searches observations on and above the Earth, in RAM or VRAM: where a device was, how sure it was, which way it looked, how far it saw, and when.
Every entry is one `usearch_geospatial_observation_t` of eleven doubles in natural units, and a field the device did not record stays NaN, which only widens the tests that read it, so a missing value must never be written as zero:

```c
usearch_geospatial_index_t photos = NULL;
usearch_geospatial_init_ram(&(usearch_geospatial_init_ram_t) {.index = &photos, .outcome = &outcome});

usearch_geospatial_observation_t const shot = {.latitude_degrees = 51.5308,
                                               .longitude_degrees = -0.1238,
                                               .altitude_metres = NAN,
                                               .accuracy_metres = 5,
                                               .heading_degrees = 10,
                                               .pitch_degrees = NAN,
                                               .heading_accuracy_degrees = NAN,
                                               .field_of_view_degrees = 65,
                                               .range_metres = 200,
                                               .begin_seconds = NAN,
                                               .end_seconds = NAN};
usearch_geospatial_insert(&(usearch_geospatial_insert_t) {.index = photos,
                                                          .executor = executor,
                                                          .outcome = &outcome,
                                                          .count = 1,
                                                          .keys = {&photo_key, sizeof(usearch_key_t)},
                                                          .observations = &shot});
```

`usearch_geospatial_search` ranks the `wanted` nearest entries per query, and `usearch_geospatial_within` reports every entry inside a radius into an arena, however many there turn out to be.
Both take a `relation`: `near_k` within the radius widened by three standard deviations of both accuracies, `visible_k` near with a clear line of sight over the handle's terrain, `sees_k` when the entry's view cone holds the query, and `seen_k` when the query's view cone holds the entry:

```c
usearch_geospatial_observation_t place = shot; // 140 m north-northeast of the camera, looking nowhere
place.latitude_degrees = 51.5320, place.longitude_degrees = -0.1233;
place.heading_degrees = NAN, place.field_of_view_degrees = NAN, place.range_metres = NAN;
usearch_matches_tape_view_t seeing = {0}; // `seeing.offsets[1]` cameras, their distances in metres
usearch_geospatial_within(&(usearch_geospatial_within_t) {.index = photos,
                                                          .executor = executor,
                                                          .outcome = &outcome,
                                                          .count = 1,
                                                          .queries = &place,
                                                          .relation = usearch_geospatial_relation_sees_k,
                                                          .tolerance = {.metres = 500},
                                                          .matches = &seeing,
                                                          .arena = arena});
```

`tolerance` carries the radius in metres, which a search may leave zero to rank by distance alone, and the slack in seconds that widens the query's time interval.
Scores are chord metres between the two positions, altitude included, which rank as great-circle arcs do and read 1 m short of the geodesic at 100 km.
A handle may test lines of sight over a `usearch_geospatial_terrain_t` given at its creation, a north-up grid of heights borrowed in place or filled by a callback, which no checkpoint carries, so every load and view names it again.
`get` and `cursor_next` return observations within the rounding the index stores them at, and the remaining verbs are the shapes ones, with `count` reporting how many entries a key holds.
The engine does not export these verbs yet, so a program calling them links against the stand-in, which tests every observation against every query, refuses every terrain and so the `visible_k` relation with `usearch_status_unsupported_k`, and refuses every VRAM verb with `usearch_status_device_k`.

## What This Binding Does Not Expose

- __Batched admission.__
  The predicate is one call per key; the span-admission form exists only in the C++ layer, since a C callback per candidate batch would cost more than it amortizes.
- __Custom metrics.__
A metric of your own, over payloads of your own, belongs to the C++ baseline tier, whose index takes any batched metric you write; see [`include/usearch/basic/README.md`](../include/usearch/basic/README.md).

## Design

- __This is the C ABI itself.__
  Every name is the one `c/usearch/` declares: a verb `usearch_dense_<verb>` takes a `usearch_dense_<verb>_t`, and closed sets are C enums with name tables, never strings.
- __The completion chooses, the verb stays.__
  A `NULL` executor runs a verb on the calling thread, an executor with the default completion blocks on it, and `usearch_completion_deferred_k` submits it, all with the same argument struct.
- __Deferred calls report through the same outcome.__
  `usearch_outcome_status` polls and `usearch_outcome_wait` blocks, both with the acquire ordering the final store pairs with.
- __No life of its own.__
  Threads live only in executors and deferred calls only in each executor's fixed table; every `*_free` waits for the calls naming its handle first.
- __Arenas are lent and reused, never shared.__
  Every output whose length scales with the call lands in an arena the caller passes; each call rewinds it, so the previous call's views go stale the moment the next one starts.
- __Nothing costs anything until it is used.__
  Blocking calls take no deferred-table slot, and an arena allocates only when a call writes into it.
- __One shape for every call.__
  Each verb takes one struct set by designator, so a call names only what it changes and the struct grows without breaking source.
- __Failures are statuses.__
  The library never logs or aborts, and a dense verb a residency lacks answers `usearch_status_unsupported_k` rather than missing from the build.
