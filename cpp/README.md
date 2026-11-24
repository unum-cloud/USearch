# USearch for C++

The C++ wrapper is a header-only layer over the C interface in `c/usearch/`, living in `cpp/usearch/` and linking the same `libusearch_shared` or `libusearch_static`.
It needs a C++20 compiler and adds no ABI of its own: every verb fills the C argument struct and calls the C function, with no layers in between.
It throws nothing, as each verb answers an `outcome` or a `deferred` call holding the library's status.

- `dense_index_t` holds fixed-width vectors under one metric, in RAM, VRAM or on NVMe.
- `sequence_index_t`, `shapes_index_t` and `geospatial_index_t` hold byte strings, 3D point clouds and observations on the Earth.
- `executor_t` owns the threads or streams a call runs on, and `arena_t` the outputs it writes.
- `outcome<T>` and `deferred<T>` hold each call's answer, blocking or submitted.

## Installation

Build the C library and add both `c/` and `cpp/` to the include path, or `include/` once installed, where the SDK places `usearch/*.hpp` beside the C headers:

```sh
cmake -B build_release -D CMAKE_BUILD_TYPE=Release -D USEARCH_BUILD_SHARED=1 -D USEARCH_BUILD_TEST=1 -D USEARCH_DENSE_DTYPES=11
cmake --build build_release --config Release --target usearch_shared usearch_test_cpp
ctest --test-dir build_release -R usearch_test_cpp
```

Consumers link `usearch::shared` after `find_package(usearch)`, as the C ones do, and include `usearch/dense_index.hpp` or the header of the modality they use.
`cpp/test.cpp` runs the same checks against the engine and against the brute-force stand-in `-D USEARCH_BUILD_STUB=1` builds.

## Quickstart

```cpp
#include <cstdio>                  // For printing.
#include <usearch/dense_index.hpp> // For the win 🚀

namespace us = unum::usearch;

int main() {
    // Threads live in the executor, the data lives in the index, and outputs land in the arena.
    auto [executor, executor_status, executor_message] = us::executor_t::for_ram();
    auto [arena, arena_status, arena_message] = us::arena_t::make({.executor = executor.handle()});
    auto [index, index_status, index_message] = us::dense_index_t::init_ram(
        {.metric = usearch_dense_metric_angular_k, .quantization = usearch_scalar_bf16_k, .dimensions = 128});
    if (index_status != usearch_status_success_k) return 1;

    usearch_key_t keys[2] {};
    keys[0].bytes[15] = 1, keys[1].bytes[15] = 2;
    float vectors[2][128] = {{1}, {0, 1}};
    if (!index.on(executor).insert<float>(keys, {&vectors[0][0], 2, 128})) return 1;

    auto [matches, status, message] = index.on(executor).search<float>({&vectors[0][0], 1, 128}, 10, arena);
    if (status != usearch_status_success_k) return 1;
    std::printf("Found %zu matches\n", matches.counts[0]);
    return 0;
}
```

Each verb names the scalar of the rows it passes, as in `insert<float>`, and answers its first output beside its status and the library's static message.

## Receivers & Deferred Calls

The receiver chooses how a verb completes, and the verb stays the same.
`index.on()` runs it on the calling thread, `index.on(executor)` blocks on an executor and answers an `outcome`, and `index.deferred(executor)` submits it there and answers a `deferred`:

```cpp
auto audited = index.on(executor).health();             // us::outcome<usearch_reachability_t>
us::deferred<usearch_reachability_t> audit = index.deferred(executor).health();
if (audit.status() == usearch_status_deferred_k) { /* other work */ }
auto [reachability, status, message] = audit.wait();     // what the blocking call would have answered
```

Both receivers are one class template over the completion, so each verb has one body that fills the C struct and lands its result in the call's output slot.
A `deferred` owns its outcome and output slot and never moves, since the library writes into it; the arena, handles, inputs and handlers it was handed stay allocated and unchanged until it is final.
Its one owner polls `status` or blocks in `wait`, and destruction waits the same way, so a scope ending is a join.
The full rules live in the [C README](../c/README.md#receivers--deferred-calls).

## Executors

One executor serves any number of indexes, and its kind must match their residency.
An empty `executor_t` runs every call on the calling thread:

```cpp
auto threads = us::executor_t::for_ram({.threads = 8});   // zero takes 7/8 of the cores
auto rings = us::executor_t::for_nvme({.threads = 4});    // one submission ring per thread
auto streams = us::executor_t::for_vram({});              // one stream per device
auto drained = threads.value.drain({.mood = usearch_mood_spinning_k});
```

One call holds an executor at a time; a second answers `usearch_status_contended_k` at once and nothing queues.
Threads that genuinely overlap take one executor apiece, while index handles stay freely concurrent.

## Arenas

Every output whose length only the call knows, like matches, monoliths, tapes and cursor batches, lands in the `arena_t` you pass, and the verb answers a view over it.
Each call rewinds its arena, so the previous call's views go stale; keep two results alive with two arenas, and give every deferred call its own.

```cpp
auto [found_arena, found_status, found_message] = us::arena_t::make({.executor = executor.handle()});
auto [fetched_arena, fetched_status, fetched_message] = us::arena_t::make({.executor = executor.handle()});
auto hits = index.on(executor).search<float>(query, 10, found_arena);
auto rows = index.on(executor).get<float>(hits.value.keys[0], fetched_arena); // `hits` stays readable
```

The arena is the verb's last required argument, before the optional config.

## Keys

A key is `usearch_key_t`, 128 bits of opaque identity, and keys need not be unique: `count` reports how many entries share one.
Spans of keys cross as `std::span<usearch_key_t const>`, so a `std::vector`, a C array or a `std::array` all fit:

```cpp
std::vector<usearch_key_t> keys(rows);
for (std::size_t row = 0; row != rows; ++row) keys[row].bytes[15] = static_cast<std::uint8_t>(row);
auto [entries, status, message] = index.on(executor).count(keys[0]);
```

## Errors

No verb throws.
Every answer holds `status` and `message`, a static string the library owns, and converts to `true` only on success, so the usual shape is a structured binding and one check:

```cpp
auto [matches, status, message] = index.on(executor).search<float>(queries, 10, arena);
if (status != usearch_status_success_k) std::fprintf(stderr, "%s: %s\n", usearch_status_name(status), message);
```

The wrapper refuses only what the C struct cannot express: a key span whose length differs from the rows, tape or observations beside it answers `usearch_status_shape_mismatch_k` before the call crosses.
Every other status is the library's own, in the order `c/usearch/*.h` lists under each verb.

## Residency

Where the payload lives is the constructor's name, as in the C lifecycle verbs, and each takes only its own hardware's placement:

```cpp
auto in_ram = us::dense_index_t::init_ram({.metric = metric, .quantization = usearch_scalar_bf16_k, .dimensions = 256});
auto in_vram = us::dense_index_t::init_vram({.metric = metric, .quantization = usearch_scalar_bf16_k, .dimensions = 256});
auto on_nvme = us::dense_index_t::view_nvme({.path = path}, rings.value);
```

A view serves a checkpoint in place and refuses every mutation with `usearch_status_read_only_k`.
The library describes where a view's memory sits, through `describe`, and the caller checks it.

## Serialization

Checkpoints ride the SafeGraphs format; a `.safegraphs` path is one monolith and a `.safegraphs.json` path a manifest of many.

```cpp
usearch_bytes_view_t const path {reinterpret_cast<std::uint8_t const*>("index.safegraphs"), 16};
auto saved = index.on(executor).save(path);
auto restored = us::dense_index_t::load_ram({.path = path}, executor);
auto metadata = us::dense_index_t::metadata_read({.path = path});
auto monolith = index.on(executor).save_buffer(arena);
auto copied = us::dense_index_t::load_ram_buffer({.buffer = monolith.value}, executor);
```

Loading blocks, as the handle is its answer, and freezes the geometry from the checkpoint's own headers.

## Dense Indexes

A dense index holds fixed-width vectors under one metric, with the dimensions, quantization and connectivity its constructor froze.
Rows cross as `rows_view<T>`, a base pointer, a count, the values per row and an optional byte stride, where a zero stride repeats the first row.
`T` names the scalar per call, so one index takes `insert<float>` and `search<us::bf16_t>` alike; the bit containers `bf16_t`, `f16_t`, the four small floats and the packed `i4x2_t`, `u4x2_t` and `u1x8_t` name formats C++ has no arithmetic type for.
Handlers are `noexcept` callables read only during one call, like the predicate of a filtered search:

```cpp
auto even = [](usearch_key_t const* keys, usearch_admission_t* admissions, std::size_t count) noexcept {
    for (std::size_t i = 0; i != count; ++i)
        admissions[i] = keys[i].bytes[15] % 2 == 0 ? usearch_admit_k : usearch_reject_k;
};
auto filtered = index.on(executor).search<float>(queries, 10, arena, {.predicate = {even}});
```

A plan keeps a refine or reshard live between calls, and its finish on the receiver consumes it:

```cpp
us::dense_refine_plan_t plan;
auto stopped = index.on(executor).refine({.progress = {cancel_at_first}, .plan = &plan});
auto resumed = index.on(executor).refine({.plan = &plan});
auto report = index.on(executor).refine_plan_finish(std::move(plan));
std::printf("Verdict: %s\n", usearch_refine_verdict_name(report.value.verdict));
```

`reshard` takes spans of source and target handles: several into one gathers, one into several cuts, and one handle as its own only source and target recuts in place.
`multiply`, `attend`, `maxsim`, cursors and search streams follow the same shape; `cpp/usearch/dense_index.hpp` documents each.

`sample` draws keys around a focus, or across the whole index, into the arena as a `us::sample_t`, each hanging from a shown or earlier draw and placed in three dimensions, so a view grows one connected sample as it zooms.
Every modality answers it alike, and `dense_metric_name` with `dense_metric_from_name` spell a metric both ways, as the sequence and shapes ones do:

```cpp
auto drawn = index.on(executor).sample(64, arena, {.seed = 42});
auto zoomed = index.on(executor).sample(64, arena, {.shown = shown, .shown_coordinates = placed, .focus = focus});
auto [metric, metric_status, metric_message] = us::dense_metric_from_name(bytes_of("angular"));
```

## Sequence Indexes

A sequence index searches byte strings, UTF-8 text, DNA, RNA and proteins under edit distances and alignments.
Sequences cross as a `tape_view<std::uint8_t>`, the bytes end to end around one more 32- or 64-bit offset than entries, as Apache Arrow lays out a binary column:

```cpp
auto [words, status, message] = us::sequence_index_t::init_ram({.metric = usearch_sequence_metric_levenshtein_utf8_k});
std::uint8_t const bytes[] = {'k', 'i', 't', 't', 'e', 'n', 's', 'i', 't', 't', 'i', 'n', 'g'};
std::uint64_t const offsets[] = {0, 6, 13};
usearch_key_t const keys[] = {key_of(1), key_of(2)};
auto inserted = words.on(executor).insert(keys, {bytes, offsets});
auto found = words.on(executor).search({bytes + 6, std::span(offsets).first(2)}, 1, arena);
auto stored = words.on(executor).get(keys[1], arena); // stored.value.entry(0) spans "sitting"
```

`count` answers a key's entries with their total and largest byte lengths.
Needleman-Wunsch takes `usearch_sequence_weights_t` and the overlap kind its `window_widths`, both in the init struct.
`sequence_refine_plan_t`, `sequence_reshard_plan_t` and the cursor work as the dense ones do.

## Shape Indexes

A shape index searches 3D point clouds of any size, like molecules and protein backbones.
Shapes cross as a `tape_view<T>` of points, `x`, `y` and `z` interleaved, with offsets counting points, and `T` names their scalar per call:

```cpp
auto [conformers, status, message] = us::shapes_index_t::init_ram(
    {.metric = usearch_shapes_metric_hausdorff_k, .quantization = usearch_scalar_f32_k});
float const atoms[5][3] = {{0, 0, 0}, {1.1f, 0, 0}, {0, 0, 0}, {1.5f, 0, 0}, {0, 1.5f, 0}};
std::uint64_t const offsets[] = {0, 2, 5}; // a 2-atom and a 3-atom entry
auto inserted = conformers.on(executor).insert<float>(keys, {&atoms[0][0], offsets});
auto shapes = conformers.on(executor).get<float>(keys[1], arena); // 9 scalars, in the inserted pose
```

`count` answers a key's entries with their total and largest point counts, and `search` takes `hints` from an earlier search.
The engine stores `f16` or `f32` points and refuses other quantizations with `usearch_status_unsupported_k`.

## Geospatial Indexes

A geospatial index searches observations on and above the Earth, each one `usearch_geospatial_observation_t` of eleven doubles, where a field no device recorded is NaN:

```cpp
auto [places, status, message] = us::geospatial_index_t::init_ram({});
usearch_geospatial_observation_t const stops[] = {
    {48.8566, 2.3522, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN},
    {48.8656, 2.3522, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN},
    {52.52, 13.405, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN}};
auto inserted = places.on(executor).insert(keys, stops);
auto nearest = places.on(executor).search({stops + 2, 1}, usearch_geospatial_relation_near_k, {}, 1, arena);
auto nearby = places.on(executor).within({stops, 1}, usearch_geospatial_relation_near_k, {.metres = 2000}, arena);
// nearby.value.view.offsets[1] == 2: both Paris stops, chords in metres
```

`search` ranks the `wanted` nearest entries the relation admits, and `within` reports every one inside the tolerance as a ragged run per query.
A terrain filled on demand is a `noexcept` handler that `us::geospatial_terrain(handler, grid)` turns into the init struct's `terrain`, which the index keeps for its whole life and calls from the executor's threads.

## What This Binding Does Not Expose

- __Exceptions.__
  No verb throws; a caller wanting them wraps `outcome` in one place of its own.

## Design

- __One body per verb.__
  The blocking and deferred receivers are one class template over the completion, so a verb fills its C struct once and the receiver's type alone picks whether it blocks.
- __The scalar is a call's type parameter.__
  Rows and point tapes name their element type per call, as in `search<float>`, never fixed by the handle.
- __Callbacks cannot throw.__
  Every handler concept requires `std::is_nothrow_invocable_r_v`, and its trampoline is `noexcept`.
- __Shared pieces, no base classes.__
  Outcomes, receivers' dispatch, plans, cursors, tapes, executors and arenas live in `usearch/types.hpp`, and each index header adds only its own verbs.
