# USearch for C++

The C++ SDK is a header-only C++20 layer over the stable C ABI in `c/usearch/`, linking the same pre-instantiated `libusearch_static` or `libusearch_shared`, so no index template is instantiated in your build.
It includes nothing but the C headers and the standard library, throws nothing, and allocates nothing on any call path.
It exposes dense, sequence, shape and geospatial indexes, and the Apache Arrow converters.
Every symbol is documented in its header under `cpp/usearch/`, and `usearch/usearch.hpp` includes them all.

## Installation

```sh
cmake -B build_release -D CMAKE_BUILD_TYPE=Release -D USEARCH_BUILD_SHARED=1 && cmake --build build_release
```

That build compiles the four most common dense storage types, `bf16`, `u1x8`, `i8` and `e4m3`, and an index over any other is refused with `status_code_t::unsupported_k`; `-D USEARCH_DENSE_DTYPES=11` compiles all eleven.
`cmake --install build_release --component sdk` places `usearch/*.hpp` beside the C headers and both libraries, and `find_package(usearch)` or a `FetchContent` or `add_subdirectory` of this repository names the SDK `usearch::cpp`.
`usearch::cpp` carries the `cpp/` include path, the C headers of `usearch::abi` and the archive `usearch::static`, and nothing of the engine's headers.

`-D USEARCH_BUILD_STUB=1` builds `usearch::static_stub`, a brute-force stand-in that answers the same verbs over the standard library alone, with exact search, seconds to build and no submodules, for testing a caller and never for shipping.
`cpp/test.cpp` runs against both, as `usearch_test_cpp` and `usearch_test_cpp_stub`.

## Quickstart

Every batch input is a `rows_view<T>` of a base pointer, a row count, the values per row and an optional byte stride, so a memory-mapped dataset feeds the index with no reshape and no copy.
An index owns entries, an `executor_t` owns the threads or streams a call runs on, and an `arena_t` holds what a call writes back.

```cpp
#include <cstdio>              // `std::printf`
#include <usearch/usearch.hpp> // `us::dense_index_t`, `us::executor_t`, `us::arena_t`, `us::deferred`

namespace us = unum::usearch;

int main() {
    auto [index, index_status] = us::dense_index_t::init_ram(
        {.metric = usearch_dense_metric_angular_k, .quantization = usearch_scalar_i8_k, .dimensions = 4});
    if (index_status.failed()) return 1;
    us::status_t const reserved = index.reserve(1000); // the calling thread; inserts grow on demand anyway

    usearch_key_t keys[2] {};
    keys[0].bytes[15] = 42;
    keys[1].bytes[15] = 43;
    float const slab[8] = {0.2f, 0.1f, 0.2f, 0.1f, 0.1f, 0.2f, 0.1f, 0.2f};
    if (reserved.failed() || index.insert<float>(keys, {slab, 2, 4}).failed()) return 1;

    auto [executor, executor_status] = us::executor_t::for_ram();
    auto [arena, arena_status] = us::arena_t::make({.executor = executor.handle()});
    auto [local_arena, local_arena_status] = us::arena_t::make();
    float const query[4] = {0.2f, 0.1f, 0.2f, 0.1f};
    auto [local, local_status] = index.search<float>({query, 1, 4}, 10, local_arena);          // the calling thread
    auto [pooled, pooled_status] = index.on(executor).search<float>({query, 1, 4}, 10, arena); // blocks on the executor
    if (local_status.failed() || pooled_status.failed()) return 1;
    us::deferred<usearch_matches_view_t> call = index.deferred(executor).search<float>({slab, 2, 4}, 10, arena);
    auto [matches, status] = call.wait(); // blocks until final; destruction waits too
    if (status.failed()) return 1;
    std::printf("Found %zu and %zu matches\n", local.counts[0], matches.counts[1]);
    return 0;
}
```

Each verb names the scalar of the rows it passes, as in `insert<float>`, and every fallible call answers its value beside a `status_t`, unpacked as `auto [value, status]`.
The index itself runs its verbs on the calling thread, and an arena made without an executor is host pages for it.

## Receivers & Deferred Calls

The receiver decides where a verb runs and how it completes, while the verb and its arguments stay the same:

- `index.search<float>(..)` runs on the calling thread.
- `index.on(executor).search<float>(..)` blocks on the executor.
- `index.deferred(executor).search<float>(..)` submits the call and answers a `deferred`.

All three are one class template over its receiver, `dense_index<receiver>`, so each verb is written once: `dense_index_t` owns the index, while `on` and `deferred` answer a `dense_index_on<completion>` naming it and the executor, both owned by the caller.
Only the owning `dense_index_t` opens, loads and hands out cursors and streams.
A blocking verb answers a `status_t` when it outputs nothing else and an `expected<T>` otherwise, so nothing about asynchrony reaches a caller who never asks for it.
Each class spells that choice once, as its `deferrable<T>`, which names a `deferred<T>` under a deferred receiver.
A `caller_owned` receiver names an index and an executor the caller owns, so neither may die while it, or a `deferred` it submitted, lives.

Any coroutine type awaits a deferred call, with any loop offering `post(void (*)(void*) noexcept, void*)`, so two shards are searched at once:

```cpp
app::task search_both(us::dense_index_t const& east, us::dense_index_t const& west,
                      us::executor_t const& east_executor, us::executor_t const& west_executor,
                      us::rows_view<float> queries, app::loop& loop) {
    auto [east_arena, east_arena_status] = us::arena_t::make({.executor = east_executor.handle()});
    auto [west_arena, west_arena_status] = us::arena_t::make({.executor = west_executor.handle()});
    us::deferred<usearch_matches_view_t> east_call = east.deferred(east_executor).search<float>(queries, 10, east_arena);
    us::deferred<usearch_matches_view_t> west_call = west.deferred(west_executor).search<float>(queries, 10, west_arena);
    auto [east_matches, east_status] = co_await east_call.wait(loop); // both shards were already running
    auto [west_matches, west_status] = co_await west_call.wait(loop);
    if (east_status.failed() || west_status.failed()) co_return;
    merge(east_matches, west_matches);
}
```

A `deferred` holds the outcome and the outputs the library writes until the call is final, and never moves, so it lives on the caller's stack or in the awaiting coroutine's frame.
The waited result is what a blocking search answers, a view into the same arena, valid until its next call.
`status` answers `status_code_t::deferred_k` while the call is not final, then its final status.
Inside a coroutine, `co_await call.wait(loop)` posts a poll to `loop` that reads the status and posts itself again until the call is final, so other tasks run meanwhile, while `call.wait()` blocks the thread without spinning.
A `loop` is any type with `post(void (*poll)(void*) noexcept, void* state)` that later runs `poll(state)` on its own thread, storing the two words; a fixed ring of them, like `test_loop_t` in `cpp/test.cpp`, is enough.
The SDK starts no thread and keeps no task, so the coroutine type and its frame are yours.
Destroying an unfinished `deferred` waits for the call.

The library reads a deferred call's inputs after the verb returns, so the keys, rows, configs, handlers and plans a deferred receiver is handed stay alive and unchanged until the call is final.
A deferred call allocates nothing: its outcome and outputs live inside the `deferred`, wherever the caller put it.

One call holds an executor at a time, so a second call finding it held finishes at once with `status_code_t::contended_k`, and nothing queues.
A deferred call on a caller-exclusive pool runs on the pool's thread 0, so polling alone finishes it, and its progress handler and predicate run on the pool's threads; an inclusive pool, a single thread, WebAssembly and a VRAM executor run it at submission.
Calls depending on each other are ordered by waiting on the first before submitting the second, since no call carries an ordering field.

## Executors

An executor owns host threads and their I/O queues, or orders work on device streams it never opens, and owns the traversal scratch each of them walks with, grown to a high-water mark.
One executor serves any number of indexes, and its residency must match the one their handles were built for.
There is a constructor per residency, each taking only what its own hardware has, and the placement lists are the ones the indexes were built with, which keeps compute and memory domains in agreement.

```cpp
std::int32_t const nodes[2] = {0, 1}, devices[2] = {0, 1};
usearch_bytes_view_t const drives[1] = {us::bytes_view("/mnt/nvme0")};

// Eight workers, where zero takes 7/8 of the cores, and threads interleaved across two NUMA nodes.
auto [threads, threads_status] = us::executor_t::for_ram({.threads = 8});
auto [domains, domains_status] = us::executor_t::for_ram({.ram = {.memory_domains = nodes, .memory_domains_count = 2}});
// Waiters and atomics no newer than Haswell.
auto [capped, capped_status] = us::executor_t::for_ram({.ram = {.cpu_class_ceiling = usearch_cpu_class_intel_haswell_k}});
auto [rings, rings_status] = us::executor_t::for_nvme({.nvme = {.paths = drives, .paths_count = 1}, .threads = 4});
auto [streams, streams_status] = us::executor_t::for_vram({.vram = {.devices = devices, .devices_count = 2}});
us::executor_t const local;                                              // owns nothing, runs on the calling thread

us::status_t const drained = threads.drain({.mood = usearch_mood_spinning_k}); // waits out the holding call, then spins
```

`for_ram` and `for_nvme` take a thread count, where zero derives one, and `for_nvme` gives every thread one submission ring.
`for_vram` takes none, since one host thread drives one stream per device, which the executor opens with itself and destroys with itself.
Your own CUDA streams in `streams` answer `status_code_t::unsupported_k` until they are wired.
A default `executor_t` owns nothing and runs every call handed to it on the calling thread.
One call holds an executor at a time, so two threads that genuinely overlap take one apiece, while index handles stay freely shared.
`drain` waits until no call holds the executor and sets the mood idle workers adopt, and destroying an executor drains it, so every arena it made is destroyed first.
`us::gpu_devices(devices)` fills a span of `usearch_gpu_device_t` and answers how many GPUs the process can drive, a device's position there being the ordinal `for_vram` names it by, while `us::version_name()` spells the linked library's version.

## Arenas

Every output whose length only the call knows, such as matches, rows, products, monoliths and cursor or stream batches, lands in an `arena_t` you pass, and the verb answers a view over it without copying.
`arena_t::make({.executor = executor.handle()})` places it where that executor computes, decided once, and `arena_t::make()` makes host pages for the calling thread.
Each call rewinds its arena, so the previous call's views die and a steady loop allocates once.
To keep a result past the next call, or to read one while making the next, pass that call another arena.
An arena serves one call at a time, so each deferred call needs its own.
Nothing in C++ retires a stale view at compile time, so a view read after its arena's next call reads whatever that call wrote.
An arena from a VRAM executor holds device memory; `arena.describe()` answers where its outputs live, and its `host_access` decides whether host code may read them: refused for CUDA and HIP device memory, readable for Apple's shared memory.
It grows, fills and frees on that executor's first stream, so it serves only calls on that executor, any other refused with `status_code_t::device_k`.
Destroy every arena before the executor that made it.

```cpp
auto [arena, arena_status] = us::arena_t::make({.executor = executor.handle()}); // make() is host pages
for (us::rows_view<float> const& batch : batches) {
    auto [matches, status] = index.on(executor).search<float>(batch, 10, arena); // rewinds: the last batch's matches are gone
    if (!status.failed()) total += matches.scores[0];
}

auto [found_arena, found_arena_status] = us::arena_t::make();     // two live results, two arenas
auto [fetched_arena, fetched_arena_status] = us::arena_t::make();
auto [found, status] = index.search<float>({query, 1, 768}, 10, found_arena);
for (std::size_t rank = 0; rank != found.counts[0]; ++rank) {
    auto [vectors, get_status] = index.get<float>(found.keys[rank], fetched_arena); // rewinds fetched, never found
    rerank(found.keys[rank], vectors.row(0));
}

task_t search_shard(us::dense_index_t const& shard, us::executor_t const& executor, us::arena_t& arena) {
    us::deferred<usearch_matches_view_t> call = shard.deferred(executor).search<float>(queries, 10, arena);
    auto [matches, status] = co_await call.wait(loop); // or: call.wait()
    if (!status.failed()) consume(matches);
}
for (std::size_t shard = 0; shard != shards_count; ++shard) // one executor and one arena per submitted call
    search_shard(shards[shard], executors[shard], arenas[shard]);
loop.run();
```

A `deferred` never moves, so many calls in flight are named locals or one coroutine frame each.
The library never frees an arena under a call: destroying one waits for the calls writing into it, and a call naming an arena that serves another call finishes with `status_code_t::contended_k` and rewinds nothing, so even an abandoned call leaves it safe.
Fixed-size reports, such as stats, descriptions and `required`, stay plain values, and so does state you carry between calls, like the attention `carry`.

## Keys

A key is `usearch_key_t`, a 128-bit identity of sixteen bytes.
Integer identities park in the low eight bytes, big-endian, so a key whose last byte is 42 prints as `00000000-0000-0000-0000-00000000002a`.

```cpp
usearch_key_t key {};
key.bytes[15] = 42;
std::span<usearch_key_t const> const keys {stored_keys}; // a std::vector, a C array or a std::array
```

Keys need not be unique: an index admits up to `values_per_key` entries under one, `count` reports how many it holds, and `get` returns them all.
The all-zero key is the nil key.

## Errors

Every fallible call answers a `status_t`, or an `expected<T>` carrying one, and a `status_t` is a `status_code_t` beside a static `note`, so nothing allocates on the failure path and nothing throws.
Recovery branches on the code, while the note names the broken invariant, step or resource for logs, falling back to the status name.
A failure the binding catches before the call crosses the boundary reports the status the engine would have used.

```cpp
auto [matches, status] = index.on(executor).search<float>(queries, 10, arena);
if (status == us::status_code_t::contended_k) retry();
else if (status.failed()) fmt::println(stderr, "{}", status.note);
```

`status_code_t` carries the C enumerators' values, so `status_code_t::deferred_k` reaches only `deferred::status`, never a final answer.
An `expected<T>` is a plain pair of `value` and `status`, unpacked as `auto [value, status]`, and a failure default-constructs the value.
The C tables spell every enumerator, as `usearch_status_name`, `usearch_scalar_name` or `usearch_dense_metric_name` do, and read them back through the `*_from_name` twins, so nothing is compared as a string.
Handlers are `noexcept` callables, since a fault cannot unwind through the library, so they report through what they capture.

## Residency

Where the payload lives is the constructor's name, as in the C lifecycle verbs, so a residency an index lacks has no constructor to call.
Each takes only the placement its own hardware has, as memory domains, devices or paths with one shard per entry, and an empty list derives the placement: every NUMA node, one free device, or wherever the manifest points.

```cpp
std::int32_t const sockets[2] = {0, 1}, gpus[4] = {0, 1, 2, 3}, key_nodes[1] = {0};
usearch_bytes_view_t const drives[1] = {us::bytes_view("/mnt/nvme0")};
usearch_ram_options_t const keys_in_ram {.memory_domains = key_nodes, .memory_domains_count = 1};

auto [in_ram, in_ram_status] = us::dense_index_t::init_ram( // one graph per socket
    {.metric = metric, .quantization = usearch_scalar_bf16_k, .dimensions = 768,
     .ram = {.memory_domains = sockets, .memory_domains_count = 2}});
auto [in_vram, in_vram_status] = us::dense_index_t::init_vram( // sharded, never replicated
    {.metric = metric, .quantization = usearch_scalar_bf16_k, .dimensions = 768,
     .vram = {.devices = gpus, .devices_count = 4}});
auto [on_nvme, on_nvme_status] = us::dense_index_t::view_nvme(
    {.path = us::bytes_view("index.safegraphs"), .nvme = {.paths = drives, .paths_count = 1}, // striped across the drives
     .ram = &keys_in_ram},                                                                    // keys resolved through RAM
    rings);
```

An NVMe handle serves a checkpoint in place and is read-only, so every mutating verb fails on it with `status_code_t::read_only_k`.
Without `ram`, it holds nothing per entry, so `count`, `get`, `sample`, MaxSim and hinted searches fail with `status_code_t::unsupported_k`.
`load_nvme` takes the same arguments and opens a manifest writable, so `refine` rewrites its parts one at a time, while a lone monolith fails with `status_code_t::unsupported_k`.
Bigger indexes come from merging, never mutating: shards build in RAM, serialize, and join a manifest.

## Serialization

The path extension picks the layout: a `.safegraphs` path is one monolith, and a `.safegraphs.json` path is a manifest cataloging many.

```cpp
usearch_bytes_view_t const path = us::bytes_view("index.safegraphs");
us::status_t const saved = index.on(executor).save(path);
auto [loaded, loaded_status] = us::dense_index_t::load_ram({.path = path});
auto [gpu, gpu_status] = us::executor_t::for_vram({.vram = {.devices = devices, .devices_count = 1}});
auto [placed, placed_status] = us::dense_index_t::load_vram({.path = path, .vram = {.devices = devices, .devices_count = 1}}, gpu);
auto [viewed, viewed_status] = us::dense_index_t::view_ram({.path = path});        // read-only, served from the mapping
auto [metadata, metadata_status] = us::dense_index_t::metadata_read({.path = path}); // geometry without loading a shard
```

Checkpoints are residency-portable: one file materializes into RAM, uploads to a device, or opens in place on disk, decided by the constructor that reads it.
The metadata carries only the four fields a checkpoint froze, so `values_per_key` and the placement stay yours to set in the load arguments.
`view_ram` answers `status_code_t::unsupported_k` until RAM shards can serve a mapping in place.
A stopped `save` leaves leftovers every reader rejects, which are yours to remove.

In-memory buffers hold single monoliths only, through `save_buffer`, `load_ram_buffer` or `load_vram_buffer`, and `metadata_read_buffer`.

```cpp
auto [arena, arena_status] = us::arena_t::make({.executor = executor.handle()});
auto [monolith, status] = index.on(executor).save_buffer(arena);
auto [restored, restored_status] = us::dense_index_t::load_ram_buffer({.buffer = monolith});
```

## Dense Indexes

A `dense_index_t` holds equidimensional dense vectors, built from `usearch_dense_init_ram_t`: `dimensions`, `metric`, the stored `quantization`, and the `connectivity` and `values_per_key` bounds, where zero derives.
Every metric derives from a single inner-product kernel with pre-computed magnitudes; arbitrary metrics belong to the C++ baseline tier, forfeiting the SIMD kernel chains and the magnitude cache both.

- `usearch_dense_metric_inner_k` - inner product.
- `usearch_dense_metric_angular_k` - cosine distance, over pre-computed magnitudes.
- `usearch_dense_metric_euclidean_k` - L2 distance.
- `usearch_dense_metric_hamming_k` - bit-level, over `u1x8` vectors alone.
- `usearch_dense_metric_jaccard_k` - bit-level, over `u1x8` vectors alone.

### Rows & Scalars

`rows_view<T>` views a packed slab the caller owns from a pointer, a row count and the values per row; a dataset whose rows are padded to a wider pitch passes the byte stride as a fourth argument, and a zero stride repeats the first row.
The scalar type of the input is independent of the stored quantization, so the `float` slab of the Quickstart lands in an `i8` index with nothing converted beforehand.
Formats C++ has no arithmetic type for are bit containers, so a call can still name them, and NumKong's types convert to them by layout.

| C++ type                               | Scalar                         |
| :------------------------------------- | :----------------------------- |
| `double`, `float`                      | `f64`, `f32`                   |
| `bf16_t`, `f16_t`                      | `bf16`, `f16`                  |
| `e5m2_t`, `e4m3_t`, `e3m2_t`, `e2m3_t` | `e5m2`, `e4m3`, `e3m2`, `e2m3` |
| `std::int8_t`, `i4x2_t`                | `i8`, `i4x2`                   |
| `std::uint8_t`, `u4x2_t`, `u1x8_t`     | `u8`, `u4x2`, `u1x8`           |

Initializing an index with a packed quantization and inserting floating-point vectors quantizes on ingest; the `u1x8_t` type also lets you pass packed bit-vectors directly.

```cpp
auto [index, index_status] = us::dense_index_t::init_ram(
    {.metric = usearch_dense_metric_hamming_k, .quantization = usearch_scalar_u1x8_k, .dimensions = 8});
us::u1x8_t const slab[2] = {{0b0000'1111}, {0b1111'0000}};
us::status_t const inserted = index.insert<us::u1x8_t>(keys, {slab, 2, 8});
```

An `insert` batch is all-or-nothing and becomes visible at once, and it overlaps safely with concurrent searches on the same handle.

### Search

Each configurable verb takes a config struct last, whose fields are named and default to what the C interface derives.
So the required arguments are the whole call, as in `search<float>(queries, 10, arena)`, `insert<float>(keys, vectors)` or `save(path)`, and a designated initializer reaches every knob.

```cpp
usearch_traversal_stats_t stats {};
auto [matches, status] = index.on(executor).search<float>(queries, 10, arena, {
    .expansion_factor = 128,
    .stats = &stats,
});
std::printf("%zu distances computed\n", stats.computed_distances);
```

The matches are padded: query `q` holds `counts[q]` keys and scores, nearest first, from `keys[q * columns]` and `scores[q * columns]`.

```cpp
for (std::size_t query = 0; query != queries.count; ++query)
    for (std::size_t rank = 0; rank != matches.counts[query]; ++rank)
        visit(matches.keys[query * matches.columns + rank], matches.scores[query * matches.columns + rank]);
```

The breadth of the candidates rides the call rather than the handle, as `expansion_factor` on every config that walks the graph, so a recall sweep is a loop over `ef` values against one immutable index, with no mutation and no synchronization between points.
`.traversal = usearch_traversal_scan_k` scores every entry instead of walking, on RAM and NVMe handles, which gives that sweep its exact ground truth; a VRAM handle refuses it with `status_code_t::unsupported_k`.
`usearch_traversal_stats_t` reports the members visited, the distances computed and the starvation `restarts`, a graph-connectivity signal rather than a cost, overwritten by every call it is passed to.

### Filtered Search

Cross-referencing an external database during traversal beats paging through results and filtering afterwards.
Non-matching entries still route the traversal, so recall does not collapse at low selectivity.

```cpp
auto is_odd = [](usearch_key_t key) noexcept { return key.bytes[15] % 2 == 1; };
auto [matches, status] = index.on(executor).search<float>({query, 1, 768}, 10, arena, {.predicate = {is_odd}});

auto in_stock = [&](std::span<usearch_key_t const> keys, std::span<usearch_admission_t> admissions) noexcept {
    inventory.lookup(keys, admissions); // one round trip per batch
};
auto [stocked, stocked_status] = index.on(executor).search<float>({query, 1, 768}, 10, arena, {.predicate = {in_stock}});
```

A predicate is a `noexcept` callable with a `const` call operator, called by the executor's threads together, so a `mutable` lambda is refused at compile time and any state it reaches is guarded.
One answering a single key as a `bool` is cheap, asked as candidates are scored.
One writing an admission per key of a batch is expensive by default, asked once per round, unless `us::predicate_t {handler, usearch_predicate_cost_cheap_k}` says otherwise.
A VRAM handle refuses a host predicate with `status_code_t::device_k`.

Every long verb's config takes a `progress` handler, called with the units finished so far and the units the call finishes if it runs to the end.
Answering `usearch_cancel_k` stops the call at the next increment boundary.

```cpp
std::chrono::steady_clock::time_point const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
auto before_the_deadline = [&](std::size_t, std::size_t) noexcept {
    return std::chrono::steady_clock::now() < deadline ? usearch_continue_k : usearch_cancel_k;
};
us::status_t const refined = index.on(executor).refine(
    {.refine = {.expansion_factor = 128}, .progress = {before_the_deadline}}).status;
if (refined.failed()) assert(refined == us::status_code_t::cancelled_k && "only the deadline stops it");
```

The handler arrives on whichever thread advanced the work and never on two at once, which is what lets it carry a counter or a deadline without a lock of its own.
Reports are denominated in whatever the verb consumes: entries for an insertion, queries for a search, a phase's entries or islands for a refine or a reshard, draws for a sample, shards for a checkpoint, and bytes for a buffer.
How often they arrive is the increment's size rather than a guarantee, so read the counts and never the number of calls.

What a stop leaves behind is each verb's own contract.
An `insert` cannot be stopped: it claims every entry before its first report, ignores a cancel, and publishes the whole batch at once.
A stopped search reports `status_code_t::cancelled_k` and leaves no matches.
A stopped `refine` or `reshard` reports `status_code_t::cancelled_k` and leaves a graph that saves; without a plan it closes the round where it stopped, and with one it leaves the plan live for the next call.
A stopped load creates no index, and a stopped `save_buffer` leaves no monolith.

### Retrieval and Streaming

`get<T>` lands every vector stored under a key in an arena, converted to the scalar type you name, one row per entry in insertion order; an absent key yields no rows.

```cpp
auto [vectors, status] = index.on(executor).get<float>(key, arena);
std::printf("%zu rows of %zu\n", vectors.written, vectors.view.columns);
auto [again, again_status] = index.on(executor).get<float>(key, arena); // rewinds it, so the vectors above are gone
```

A cursor walks the index contents, reading each shard's publication watermark as it enters that shard, so an entry once passed is never revisited.
Without start keys it yields every entry as stored; with them, the starts' own entries first, then the entries those link to breadth-first, each once.
`cursor_next` walks the key column alone, or copies the keys each entry links to beside it when asked for links, both into an arena; the vectors themselves come from `get`.

```cpp
auto [cursor, cursor_status] = index.cursor_init();
for (;;) {
    auto [batch, status] = index.on(executor).cursor_next(cursor, 1024, arena, true);
    if (status.failed() || batch.written == 0) break;
    std::printf("%zu keys, the first linking to %zu\n", batch.written, batch.links.offsets[1] - batch.links.offsets[0]);
}
```

A search stream is a resumable best-first traversal, so callers that post-filter aggressively pull until satisfied instead of guessing "k" upfront.
It walks exactly one query row, and each pull lands up to `wanted` more matches in an arena as the one query of `usearch_matches_view_t`, empty once the walk is drained.
No residency implements the stream yet, so the engine answers `status_code_t::unsupported_k`.

```cpp
auto [stream, stream_status] = index.search_stream_init<float>({query, 1, 768});
for (bool pulling = !stream_status.failed(); pulling;) {
    auto [pulled, status] = index.on(executor).search_stream_next(stream, 64, arena);
    pulling = !status.failed() && pulled.counts[0] != 0;
    for (std::size_t rank = 0; pulling && rank != pulled.counts[0]; ++rank) {
        if (pulled.scores[rank] > 0.3) pulling = false;
        else visit(pulled.keys[rank], pulled.scores[rank]);
    }
}
```

### Refinement

`refine` re-runs the neighbor selection over entries already in the graph and answers a `usearch_refine_report_t` of the nodes re-selected, those whose neighbors `changed`, the islands found and reconnected, the neighborhoods re-stitched with the links added and dropped, the `verdict`, and the `opening` and `reachability` health checks around the round, so a batch built at a low expansion factor reaches the quality of a slower build afterwards.
A `verdict` other than `usearch_refine_improved_k` is what settled looks like.
On a RAM handle searches may run throughout, on a VRAM one they may not, and insertions never may.

```cpp
us::dense_index_on<usearch_completion_blocking_k> const receiver = index.on(executor);
for (;;) {
    auto [report, status] = receiver.refine({.refine = {.expansion_factor = 128}});
    if (status.failed() || report.verdict != usearch_refine_improved_k) break;
}
```

On a loaded NVMe handle `memory_limit` bounds the bytes a round holds for the one part it rewrites: zero rounds the part in RAM, and a smaller budget streams its rounds from the drives.

A `dense_refine_plan_t` carries a round across calls: a stop leaves it live, the next call holding it continues where it stopped, `phase` reports the phase the call stopped in, and `refine_plan_finish` closes it.

```cpp
us::dense_refine_plan_t plan;
while (receiver.refine({.progress = {before_the_deadline}, .plan = &plan}) == us::status_code_t::cancelled_k)
    std::printf("stopped in %s at %zu of %zu\n", usearch_phase_name(plan.phase().phase), plan.phase().completed,
                plan.phase().total);
auto [report, status] = receiver.refine_plan_finish(std::move(plan));
```

A deferred call reads and writes the plan it carries, which stays where it is until the call is final.

```cpp
us::deferred<usearch_refine_report_t> call = index.deferred(executor).refine({.plan = &plan});
auto [answer, status] = co_await call.wait(loop); // or: call.wait()
```

### Resharding

`reshard` moves the entries of its `sources` into its `targets`, keeping their links rather than re-inserting their entries, then runs a round on every target that took entries; the receiver supplies the executor.
Several sources gather into the one target among them, one source cuts into several empty targets, and one handle alone recuts its own shards in place under `shard_capacity`.
Every handle stays yours: a source the targets do not name ends empty and live.
Readers of a RAM target may keep searching throughout, while a VRAM one admits no overlapping search.

```cpp
usearch_dense_index_t const gathered[3] = {index.handle(), first_shard.handle(), second_shard.handle()};
us::status_t const gather = index.on(executor).reshard(gathered, {gathered, 1}).status;
usearch_dense_index_t const source[1] = {index.handle()}, parts[2] = {left.handle(), right.handle()};
us::status_t const cut = index.on(executor).reshard(source, parts).status; // two empty targets take a part each

```

A stopped `reshard` cuts only its closing rounds, every entry already moved, and reports `status_code_t::cancelled_k`; a `dense_reshard_plan_t` continues it as a refine plan does, and `reshard_plan_finish` closes it.

### Sampling

`sample` draws keys around a focus, each hanging from a shown or earlier draw, and places them in three dimensions beside the keys already shown, so a view grows one connected sample as it zooms.

```cpp
auto [drawn, status] = index.on(executor).sample(256, arena, {
    .shown = shown,
    .shown_coordinates = {&xyz[0][0], shown.size(), 3},
    .focus = near,
    .seed = 42,
});
for (std::size_t draw = 0; draw != drawn.written; ++draw) place(drawn.keys.data[draw], drawn.coordinate(draw));
```

### Sparse Multiplication

`multiply`, `maxsim` and `attend` treat the index as a sparse operand rather than as a kNN store, and each folds its arithmetic into the graph walk instead of materializing a match list.
All three need logits rather than distances, so they run on an inner-product index alone.

`multiply` is a sparse matrix product: every left row is walked against the index, and every pair the walk scored lands in an arena as one ragged run, in traversal order.
`wanted` keeps the first pairs of each row, where zero keeps every pair, and `required` counts how many there were.
The run's Apache Arrow offsets land in the arena beside the keys and products.

```cpp
auto [products, status] = index.on(executor).multiply<float>({left, rows, 768}, arena, {.wanted = 32});
for (std::size_t row = 0; row != rows; ++row)
    std::printf("%zu pairs\n", products.view.offsets[row + 1] - products.view.offsets[row]);
```

### Late Interaction

`maxsim` is late interaction: every query row takes its best match within a key, and those maxima sum to that key's score.
A document is one key holding its token rows, which is what a `values_per_key` above one exists for, and the wanted count is over documents rather than rows.
Each query lands its best documents, largest score first, in the same ragged `dense_products_t` run `multiply` answers, and a `wanted` of zero keeps every document the query scored.

```cpp
auto [documents, status] = index.on(executor).maxsim<float>({tokens, rows, 768}, tokens_per_query, 10, arena);
for (std::size_t query = 0; query != rows / tokens_per_query; ++query)
    std::printf("%zu documents\n", documents.view.offsets[query + 1] - documents.view.offsets[query]);
```

### Attention

`attend` is Flash Attention folded into the walk: every key it scores folds into each head's running maximum, partition and weighted value sum, with no top-k cut, and the outputs land in an arena already divided by the partition, as rows of the scalar type you name.
The index stores no values, so a `value_lookup_t` handler resolves every scored key to its value row, as wide as the keys, stored in the index's own quantization, and read in place from storage the caller keeps until the call is final.
A null row, or rows of another scalar type or width than the library reads, refuses the call with `status_code_t::shape_mismatch_k`.
It runs on the executor's threads together.
A VRAM walk resolves values on the device, where no host handler runs, so `value_lookup_t::device` names a device function and its state instead, and a host handler with an arena from a VRAM executor is refused with `status_code_t::device_k`.
With a `head_count` of 8, row `t * 8 + h` of the queries is head `h` of token `t`, and the heads of one token share its walk.
A `carry` of one `usearch_dense_softmax_state_t` per query row, owned by you, folds this branch into another over the same queries, such as a recent window kept out of the index.
`score_scale` is conventionally the inverse square root of the head width, and zero derives it from the frozen geometry.

```cpp
auto value_lookup = [&](usearch_key_t const* keys, std::size_t count, usearch_scalar_t, std::size_t,
                        void const** rows) noexcept {
    for (std::size_t position = 0; position != count; ++position) rows[position] = cache.find(keys[position]);
};
std::vector<usearch_dense_softmax_state_t> carry(queries.count);
auto [context, status] = index.on(executor).attend<float, float>(
    queries, value_lookup, arena, {.head_count = 8, .carry = carry.data()});
std::printf("%zu rows of %zu\n", context.written, context.view.columns);
```

### Introspection

```cpp
auto [description, status] = index.describe();
std::printf("%zu entries in %zu bytes\n", description.size, description.memory_usage);
std::printf("dispatched to %s and %s\n", usearch_cpu_class_name(description.cpu_class),
            usearch_gpu_class_name(description.gpu_class));
std::printf("compiled for %s\n", usearch_capabilities_compiled());
std::printf("available here %s\n", usearch_capabilities_available());

auto [reachability, health_status] = index.health();
std::printf("%zu of %zu entries reachable\n", reachability.reachable, reachability.audited);

auto [entries, count_status] = index.count(key);
std::printf("%zu entries under the key\n", entries);
```

`describe` reads the frozen metadata and the counters once each, so under concurrent insertions they may come from slightly different moments, and `memory_usage` excludes the executor's scratch.
`health` measures how much of the graph a search reaches from the root; entries it misses stay unfound until a `refine` re-links them.

## Sequence Indexes

A `sequence_index_t` searches byte strings, UTF-8 text, DNA, RNA and proteins under edit distances and alignments.
Its sequences cross as one `tape_view<std::uint8_t>`, the Apache Arrow `Binary` layout of bytes end to end around one more 32- or 64-bit offset than entries, so the two runs of an Arrow column pass as they are.

```cpp
auto [index, index_status] = us::sequence_index_t::init_ram({.metric = usearch_sequence_metric_levenshtein_utf8_k});
std::uint8_t const words[] = {'k', 'i', 't', 't', 'e', 'n', 's', 'i', 't', 't',
                              'i', 'n', 'g', 'm', 'i', 't', 't', 'e', 'n'};
std::uint64_t const offsets[] = {0, 6, 13, 19};
us::status_t const inserted = index.insert(keys, {words, offsets});

std::uint8_t const probe[] = {'s', 'i', 't', 't', 'i', 'n'};
std::uint64_t const probe_offsets[] = {0, 6};
auto [arena, arena_status] = us::arena_t::make();
auto [matches, status] = index.search({probe, probe_offsets}, 1, arena);
assert(matches.scores[0] == 1); // one edit away
```

`usearch_sequence_metric_needleman_wunsch_k` takes `weights`, a 256 × 256 substitution matrix with a gap cost, and `usearch_sequence_metric_overlap_k` takes `window_widths`; every load and view names the weights again, since no checkpoint records them.
`get` writes a key's sequences into an arena as `sequence_entries_t`, whose `entry(position)` spans one sequence's bytes, and `count` reports a key's entries with their total and largest byte lengths.
The receivers, reports and plans work as the dense ones do: `refine` and `reshard` take a `sequence_refine_plan_t` or a `sequence_reshard_plan_t` and answer a `usearch_refine_report_t`.

## Shape Indexes

A `shapes_index_t` searches 3D point clouds of any size, like molecules, conformers and protein backbones, invariant to rotation, reflection, translation and point order.
Its tape holds `[x, y, z]` points end to end, with offsets that count points, as an Arrow `List` column lays them out:

```cpp
auto [index, index_status] = us::shapes_index_t::init_ram( // Chamfer over f32 points
    {.metric = usearch_shapes_metric_chamfer_k, .quantization = usearch_scalar_f32_k});
float const points[5][3] = {{0, 0, 0}, {1.1f, 0, 0}, {0, 0, 0}, {1.5f, 0, 0}, {0, 1.5f, 0}};
std::uint64_t const offsets[] = {0, 2, 5}; // a pair and a triple
us::status_t const inserted = index.insert<float>(keys, {&points[0][0], offsets});

float const probe[2][3] = {{0, 0, 0}, {1, 0, 0}};
std::uint64_t const probe_offsets[] = {0, 2};
auto [arena, arena_status] = us::arena_t::make();
auto [matches, status] = index.search<float>({&probe[0][0], probe_offsets}, 1, arena);
```

The metrics are Chamfer, Hausdorff, Gromov-Wasserstein and Wasserstein-Procrustes, over `f16` or `f32` storage, whatever scalar the points arrive in.
A non-zero `points` in `usearch_shapes_init_ram_t` freezes every shape's point count.
`get<T>` converts each shape to the scalar it is asked for and answers it in its inserted pose, and `count` reports a key's total and largest point counts.

## Geospatial Indexes

A `geospatial_index_t` searches observations on and above the Earth, like photos, video frames, drone footage and sensor fixes.
`usearch_geospatial_observation_t` is the library's own eleven doubles, so a span of them crosses with no copy; a field no device recorded stays NaN, which only widens the tests that read it:

```cpp
double const unrecorded = std::numeric_limits<double>::quiet_NaN();
auto at = [=](double latitude, double longitude) noexcept {
    return usearch_geospatial_observation_t {latitude,   longitude,  unrecorded, unrecorded, unrecorded, unrecorded,
                                             unrecorded, unrecorded, unrecorded, unrecorded, unrecorded};
};
auto [index, index_status] = us::geospatial_index_t::init_ram({});
usearch_geospatial_observation_t const stops[3] = {at(51.5308, -0.1238), at(51.5320, -0.1233), at(51.5033, -0.1196)};
us::status_t const inserted = index.insert(keys, stops);

auto [arena, arena_status] = us::arena_t::make();
auto [nearby, status] = index.within({stops, 1}, usearch_geospatial_relation_near_k, {.metres = 500}, arena);
assert(nearby.view.offsets[1] == 2); // both stops within 500 metres, in chord metres
```

`within` reports every entry the relation admits inside the radius, however many, while `search` ranks the `wanted` nearest, a zero `metres` ranking by distance alone.
The relation is an argument of each call: `near` widens the radius by both positions' accuracies, `sees` and `seen` test the view cones, and `visible` tests a line of sight over the terrain the handle was built with, which `us::geospatial_terrain(handler, grid)` fills on demand.
The index keeps that handler for its whole life and calls it from the executor's threads at once, so its call operator is `const`.
`get` answers a key's observations within the stored rounding.

## Apache Arrow

The converters of `c/usearch/arrow.h` view an Arrow column as a verb's input in place, and export what a call wrote into an arena as a column in place, living until the arena's next call.
`usearch/arrow.hpp` wraps each of them in a `noexcept` function answering a `status_t`, with the views beside an entry count:

```cpp
#include <usearch/arrow.hpp>

auto [vectors, get_status] = index.get<float>(key, arena);
ArrowSchema schema {};
ArrowDeviceArray array {};
if (us::rows_view_to_arrow(vectors.view, vectors.written, arena, schema, array).failed()) return 1;

auto [column, view_status] = us::rows_view_from_arrow(schema, array.array); // the same rows, in place
us::rows_view<float> const queries {static_cast<float const*>(column.view.data), column.count, column.view.columns,
                                    column.view.stride};
auto [matches, status] = index.search<float>(queries, 10, other);
schema.release(&schema); // schema and array release apart
array.array.release(&array.array);
```

`keys_view_from_arrow`, `rows_view_from_arrow` and `tape_view_from_arrow` read keys, rows and sequence or shape tapes, and `keys_view_to_arrow`, `rows_view_to_arrow`, `tape_view_to_arrow`, `keys_tape_view_to_arrow` and `matches_tape_view_to_arrow` write keys, rows, tapes, a cursor's links and ragged matches.

## Left Unbound

These C entry points also take a deferred completion, but the SDK calls them blocking, whatever receiver you hold:

- The `load_*` family, `view_nvme`, `load_nvme`, `metadata_read` and `metadata_read_buffer`, with each one's `executor` argument choosing where it fans out.
- Cursor and search-stream openings, which name no executor, and their pulls, which run on the receiver's.

`init_ram`, `init_vram` and `view_ram` block in C as well.
No verb throws, so a caller wanting exceptions wraps `status_t` in one place of its own.

## Design

- __A thin layer over the C ABI.__
  Every name follows `c/usearch/`: a C verb `usearch_dense_<verb>` is a `<verb>` here, each header mirrors one C header, and a verb's argument struct becomes its `dense_*_config_t` or is taken as is.
  Enums are the C enums, never strings.
- __The receiver chooses, the verb stays.__
  The index runs a verb on the calling thread, `on(executor)` blocks on an executor, and `deferred(executor)` submits it, all with the same arguments.
- __Deferred calls use the language's own idioms.__
  `status` polls the status word, `wait()` blocks on it, and `co_await call.wait(loop)` suspends and re-polls through the caller's scheduler, as Rust's `Deferred` re-wakes itself through its `RawWaker`, since the library has no wake-up.
- __No life of its own.__
  The SDK starts no threads, keeps no queues and caches nothing; destroying a `deferred` waits for the call first.
- __Arenas are the caller's, reused, never shared.__
  Every output whose length scales with the call lands in an arena the caller passes, and the answer is a view over it.
- __Nothing costs anything until it is used.__
  A blocking call allocates nothing, keeping its outcome and outputs on the stack; a deferred submission allocates nothing, keeping them in the immovable `deferred` the caller holds; an awaiter lives in the coroutine frame, and the scheduler stores two words, a function and a state pointer.
- __Callers pass memory, never an allocator.__
  No allocation callback crosses into the library: payload belongs to the index, placed by its constructor's residency, and scratch belongs to the executor.
- __Callbacks cannot throw.__
  Every handler concept requires a `noexcept` call, and its trampoline is `noexcept`.
- __Shared pieces, no base classes.__
  Statuses, expected values, deferred calls, owners, plans, tapes, executors and arenas live in `usearch/types.hpp`, held as members, and each index header adds only its own verbs.

| Path | C++ | Rust |
|---|---|---|
| Blocking call | 0 | 0 |
| Refusal before the call | 0 | 1 error |
| Deferred submit and wait | 0: the `deferred` lives with the caller | 0: the `Deferred` is pinned where it is polled |
| Each callback | 0 | 0 |
