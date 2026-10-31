# USearch for Wolfram

USearch binds the C ABI in `include/usearch/` to the [Wolfram Language](https://www.wolfram.com/language/) through `LibraryLink`, with `wolfram/lib.c` as the boundary and the `USearch.wl` package as the interface.
It needs Mathematica or the Wolfram Engine 12 or newer, on Linux, macOS or Windows, since keys and monoliths cross as `NumericArray` and lists as `DataStore`.
It exposes dense indexes; sequence, matrix and geospatial indexes follow as the C library ships them.

## Installation

The library is a subdirectory of the root project rather than a project of its own, so build it from the repository root:

```sh
cmake -B build_release -D USEARCH_BUILD_WOLFRAM=1 -D USEARCH_DENSE_DTYPES=11 \
    -D USEARCH_WOLFRAM_PATH=/usr/local/Wolfram/Mathematica
cmake --build build_release --config Release --target usearch_wolfram

# Resulting artifact will be named like:
#   build_release/usearch_wolfram.so     on Linux
#   build_release/usearch_wolfram.dylib  on macOS
#   build_release/usearch_wolfram.dll    on Windows
```

Set `USEARCH_WOLFRAM_PATH` to your installation when it is not `/usr/local/Wolfram/Mathematica`; the SDK contributes only the `LibraryLink` headers under it.
`USEARCH_DENSE_DTYPES=11` compiles every dense storage type; the default four, `bf16`, `u1x8`, `i8` and `e4m3`, build faster but refuse the other seven.
`usearch_wolfram` is the one library here without a `lib` prefix, because Wolfram finds a library dropped into `LibraryResources/$SystemID` by its bare name.

The checks load the package against a built library and exit non-zero at the first failure:

```sh
wolframscript -file wolfram/Test.wls [path/to/usearch_wolfram] # defaults to build_release/usearch_wolfram.so
```

## Quickstart

Put the library's directory on `$LibraryPath` and load the package.
Below is a minimal example mirroring the API in `wolfram/Test.wls`.

```wolfram
PrependTo[$LibraryPath, "build_release"];
Get["wolfram/USearch.wl"];

(* Create an index: metric, quantization, dimensions *)
index = DenseInitRam["inner", "bf16", 3];
DenseReserve[index, 10];

(* Insert a batch, one key per vector row *)
rows = {{0.2, 0.6, 0.4}, {0.4, 0.6, 0.2}};
DenseInsert[index, {FromDigits["5f9c2b718e4d4a63b2f10c7a9d3e6b58", 16], 43}, rows];

(* Search on the kernel's own thread, five matches per query, written in an arena *)
local = ArenaInit[];
matches = DenseSearch[index, rows, 5, local]
matches["Keys"]       (* {{key, key}, {key, key}}, nearest first *)
matches["Scores"]

(* Spawn the workers the same verbs run on, and an arena where they compute *)
executor = ExecutorForRam[];
arena = ArenaInit["Executor"->executor];
matches = DenseSearch[DenseIndexOn[index, executor], rows, 5, arena];

(* Get and multiply reuse the same arena, and every result is copied out of it *)
DenseGet[index, 43, arena]
products = DenseMultiply[DenseIndexOn[index, executor], rows, arena];

(* Metadata and counters *)
DenseDescribe[index]

(* Save, load as a new index; handles are freed once nothing refers to them *)
DenseSave[DenseIndexOn[index, executor], "index.safegraphs"];
loaded = DenseLoadRam["index.safegraphs", "Executor"->executor];
ClearAll[loaded, index, arena, local, executor];
```

## Receivers

The receiver chooses where a verb runs, and the verb stays the same.
A `DenseIndex` runs it on the kernel's own thread, and `DenseIndexOn[index, executor]` fans it out over an executor's workers:

```wolfram
matches = DenseSearch[index, queries, 10, arena];                            (* the kernel's thread *)
matches = DenseSearch[DenseIndexOn[index, executor], queries, 10, arena];    (* the executor's workers *)
```

Either way the call blocks and returns once its outcome is final, so a notebook never polls, waits or orders calls.
The C interface's deferred calls are left out of this binding.
Constructors address no index yet, so they take no receiver; the ones the C interface lets fan out take an `"Executor"` option instead.

## Executors

An executor owns the host threads or device streams; an index owns the entries.
One executor serves several indexes, and its residency must match theirs.
There is a constructor per C one, each taking every argument its own hardware has as an option, where zero and an empty list derive:

```wolfram
ExecutorForRam["MemoryDomains"->{0, 1}, "Threads"->0]
ExecutorOnPool[poolAddress, "MemoryDomains"->{}]                     (* borrows a fu_pool_t *)
ExecutorForVram["Devices"->{0}, "Streams"->{}, "ScratchLimit"->0]
ExecutorForNvme["Paths"->{"/dev/nvme0n1"}, "Threads"->0]
ExecutorDrain[executor, "parked"]                                    (* or "spinning" *)
```

`ExecutorDrain` picks whether the workers sleep or spin once idle, `parked` when the mood is left out.
`"Streams"` lists borrowed CUDA stream addresses, one per device, and an empty list makes the executor create its own.

## Arenas

Every output whose length only the call knows, such as matches, rows, products and monoliths, lands in an arena you pass.
`ArenaInit["Executor"->executor]` places it where that executor computes, decided once, and `ArenaInit[]` makes host pages for the kernel's own thread.
Each call rewinds its arena and copies its outputs into Wolfram arrays, so a steady loop allocates once and every result outlives the next call.
An arena serves one call at a time, so a second concurrent call fails as `contended`.
LibraryLink cannot alias memory it did not allocate, so the results stay valid after the arena is released too.
An arena from a VRAM executor holds device memory, which these functions refuse with `device`.

```wolfram
arena = ArenaInit["Executor"->executor];
total = 0;
Do[
    matches = DenseSearch[DenseIndexOn[index, executor], batch, 10, arena];    (* rewinds, allocates once *)
    total += Total[Flatten[matches["Scores"]]],
    {batch, batches}];
found = DenseSearch[index, {query}, 10, arena];
vectors = DenseGet[index, #, arena] & /@ First[found["Keys"]];               (* found is a copy, so one arena serves both *)
```

The arena is the last required argument, options follow it, and a call without one stays unevaluated.
Fixed-size reports such as descriptions, audits and counts stay plain values and take no arena.

## Keys

Keys are USearch's 128-bit identities, written as integers below 2¹²⁸ and crossing as sixteen bytes, most significant first.
That is what `IntegerDigits[key, 256, 16]` renders and `FromDigits[bytes, 256]` reads back, so an identity crosses at full precision whether it started as a machine integer, a UUID, or a number too wide for either.
An integer identity is the encoding of a number below 2⁶⁴, so it parks in the low eight bytes with the rest zeroed.
A UUID is the number its hexadecimal digits spell:

```wolfram
uuid = FromDigits[StringDelete["5f9c2b71-8e4d-4a63-b2f1-0c7a9d3e6b58", "-"], 16];
DenseInsert[index, {uuid, 43}, rows];
```

## Errors

A failing call answers `Failure[status, <|"MessageTemplate"->detail|>]`, where the status is the C name `usearch_status_name` spells, such as `shape-mismatch`, `io` or `read-only`, and the detail the engine's message:

```wolfram
result = DenseInsert[index, {44}, {{1.0, 2.0}}];
If[FailureQ[result], result[[1]]] (* "shape-mismatch" *)
result["MessageTemplate"] (* "The rows' width differs from the frozen geometry" *)
```

Arguments the verb's types do not admit fail as `invalid-argument`, and a call naming a released handle by its ID fails as `uninitialized`.

## Residency

Where the payload lives is the constructor's name, as in the C lifecycle verbs, so a residency an index lacks has no constructor to call.

```wolfram
DenseInitRam["inner", "bf16", 768, "MemoryDomains"->{0, 1}, "Connectivity"->0, "ValuesPerKey"->0]
DenseInitVram["inner", "bf16", 768, "Devices"->{0, 1, 2, 3}]
DenseLoadRam["index.safegraphs", "Executor"->executor, "MemoryDomains"->{}, "ValuesPerKey"->0]
DenseLoadVram["index.safegraphs", "Executor"->gpu, "Devices"->{0}]
DenseViewRam["index.safegraphs"]                                   (* read-only, served from the mapping *)
DenseViewNvme["index.safegraphs", "Executor"->drives, "Paths"->{"/dev/nvme0n1"}]
```

`DenseInitRam[dimensions]` makes an `angular` `bf16` index.
A viewed index is read-only, and every mutating verb on it fails as `read-only`.
Until NVMe checkpoints carry keys, `DenseViewNvme` fails as `unsupported`.

## Serialization

`DenseSave[receiver, path]` writes a checkpoint, and the path carries the SafeGraphs extension that picks the layout: `.safegraphs` for one monolith, `.safegraphs.json` for a manifest beside its shards.
`DenseMetadataRead[path]` answers the frozen geometry, the metric, quantization, dimensions and connectivity, without loading a shard.
In-memory buffers hold single monoliths only, as `UnsignedInteger8` numeric arrays, and the loaders take a `ByteArray` too:

```wolfram
monolith = DenseSaveBuffer[index, arena];
restored = DenseLoadRamBuffer[monolith];
DenseMetadataReadBuffer[monolith]
```

`DenseLoadVramBuffer` takes the same options as `DenseLoadVram`.

## Dense Indexes

### Rows & Scalars

Packed real lists cross as `f64`; `NumericArray` inputs keep their `Real32` or `Real64` storage.
LibraryLink borrows both as read-only `"Constant"` arguments for the blocking call, so an existing packed list or typed array needs no input copy at this boundary.
The engine still converts the input into the index's quantization.

```wolfram
vectors = NumericArray[{{1., 2., 3.}, {3., 2., 1.}}, "Real32"];
DenseInsert[index, {41, 42}, vectors];
DenseSearch[index, vectors, 2, arena]
```
A matrix is a batch of rows, so a whole batch crosses in one call rather than one crossing per vector, and `DenseInsert` and `DenseMultiply` also take a single vector as a batch of one.
The input's own width crosses with the rows, and the engine refuses one that differs from the frozen geometry as `shape-mismatch`.
Metrics are `inner`, `angular`, `euclidean`, `hamming`, and `jaccard`; quantizations are `bf16`, `f16`, `e5m2`, `e4m3`, `e3m2`, `e2m3`, `i8`, `i4x2`, `u8`, `u4x2`, and `u1x8`.

- `DenseReserve[receiver, capacity]` reserves the total capacity, including the current size, and `DenseClear[receiver]` erases every entry, keeping it.
- `DenseInsert[receiver, keys, vectors]` admits one row per key and fails as `shape-mismatch` when the counts differ.

No `NumericArray` holds zero keys, so `DenseInsert[receiver, {}, {}]` answers `Null` in the package without reaching the library, as the C call would.

### Search

`DenseSearch[receiver, queries, wanted, arena]` answers `<|"Keys"->…, "Scores"->…|>`, one list per query, holding the matches it found, nearest first.
A query that finds fewer than `wanted` matches is not an error, and its lists are just shorter.
No tensor holds zero rows either, so a search with no queries answers `<|"Keys"->{}, "Scores"->{}|>` in the package.

The breadth of the candidates rides each call in the C interface rather than the handle, so it is an `"ExpansionFactor"` option on every verb that walks the graph, zero deriving it.
`"Traversal"->"scan"` skips the walk and scores every entry exactly, on RAM and NVMe handles:

```wolfram
DenseSearch[index, queries, 10, arena, "ExpansionFactor"->64]
DenseSearch[index, queries, 10, arena, "Traversal"->"scan"]
```

### Retrieval

`DenseGet[receiver, key, arena]` answers every row stored under a key as a `rows × dimensions` real matrix, empty for an absent key.

### Refinement

`DenseRefine[receiver]` re-links the graph and answers how many nodes had their neighbors changed, taking `"ExpansionFactor"` like the other walks.
It is what makes the entries `DenseHealth` finds unreachable searchable again.

### Sparse Multiplication

`DenseMultiply[receiver, left, arena]` runs on the `inner` metric and answers `<|"Offsets"->…, "Keys"->…, "Scores"->…, "Required"->…|>`.
Row `i` of the left matrix owns pairs `offsets[[i]] + 1` through `offsets[[i + 1]]`, in traversal order, and `"Required"` counts the pairs scored.
`"Wanted"->n` keeps at most that many pairs per left row, zero keeping every pair the walk scored.
Any other metric fails as `config-mismatch`.

Pass an ordered list of column keys to get a Wolfram `SparseArray`:

```wolfram
products = DenseMultiply[index, left, {42, 41}, arena];
products . right
```

Column 1 names key 42 & column 2 names key 41, including full-width UUID keys.
Each stored product is a dot product against a quantized indexed vector; unvisited entries are zero, so this approximates a dense product rather than multiplying arbitrary sparse inputs exactly.
Column keys must be unique & cover every returned key; repeated keys within a product row are refused because they cannot identify distinct matrix columns.
`"Wanted"` truncates traversal order, rather than selecting the largest products.
The original three-argument form keeps the offsets & keys, which identify the computed entries even when a computed score is zero.

### Late Interaction

`DenseMaxsim[receiver, queries, rowsPerQuery, wanted, arena]` scores multi-row queries late-interaction style, answering `<|"Offsets"->…, "Keys"->…, "Scores"->…, "Required"->…|>` as `DenseMultiply` does.
Query `i` owns keys `offsets[[i]] + 1` through `offsets[[i + 1]]`, largest summed maxima first, and `"Required"` counts the keys scored before `wanted` cut any.
`wanted` keeps at most that many keys per query, zero keeping every key a query scored.
The token rows of every query are laid end to end, so their number divides by `rowsPerQuery`.

### Folded Attention

Prepare values once in the index's quantization, then reuse them across attention calls:

```wolfram
values = DenseValues[index, {41, 42}, NumericArray[valueRows, "Real32"]];
attention = DenseAttend[index, queryRows, values, arena,
    "HeadCount"->2, "ExpansionFactor"->64, "ScoreScale"->0.1];
attention["Outputs"]
attention["Carry"]
```

`DenseValues` stores sorted 128-bit keys & quantized rows in kernel-owned `NumericArray` buffers.
Values have the same width as the indexed keys, with one value row per unique key; every key the walk scores needs a value.
Native worker threads look up those immutable rows directly, without entering the evaluator.
This cache serves RAM & NVMe attention; device value lookup is not exposed here.

`DenseAttend` folds every scored key-value pair into online softmax, including candidates rejected by the nearest-neighbor candidates.
It requires the `inner` metric, as `DenseMultiply` does.
`"HeadCount"` consecutive query rows share one key-value head & graph walk; it must divide the number of query rows.
Zero means one head.
`"ScoreScale"->0.` derives the usual inverse square root of the dimension.
The answer contains `"Outputs"`, one real row per query head, & `"Carry"`, one `{maximumLogit, partition}` row per head.
The partition is the sum of exponentials relative to that maximum, over the scored subset.
Native folding accumulates in `f32`; the binding exports real tensors as `f64`.
Outputs & carry are copied before the Arena is reused.

### Introspection

- `DenseDescribe[receiver]` answers the metric, quantization, dimensions, connectivity, size, capacity, memory usage and dispatched capabilities.
- `DenseHealth[receiver]` answers `<|"Audited"->…, "Reachable"->…, "Sweeps"->…|>`: the entries published when the audit started, how many of them a search reaches, and the levels the sweep took to converge.
- `DenseCount[receiver, key]` counts the entries under one key.
- `VersionName[]`, `CapabilitiesCompiled[]` and `CapabilitiesAvailable[]` describe the library itself.

## What This Binding Does Not Expose

- Deferred calls, since every call blocks.
- Search predicates and progress callbacks, since the kernel evaluates on one thread and the executor's workers would have nowhere to call back into.
- Traversal telemetry, cursors, search streams & `absorb`, which the C interface has and this package does not bind yet.

## Design

- __A thin layer over the C ABI.__
  Every name follows `include/usearch/`: a C verb `usearch_dense_<verb>` is a `Dense<Verb>` here, every option is a C field, and every usage message names the C symbol it calls or says it is binding-only.
  Metrics, quantizations, moods and statuses cross as the strings `usearch_*_name` spells, read back through `usearch_*_from_name`, so the C library owns every spelling.
- __The receiver chooses, the verb stays.__
  The index runs a verb on the kernel's own thread, and `DenseIndexOn[index, executor]` runs it on an executor's workers, with the same arguments.
- __No life of its own.__
  The package starts no threads and keeps no queues, so every worker belongs to an executor you made.
- __Arenas are lent and reused, never shared.__
  Every output whose length scales with the call lands in an arena the caller passes, and is copied into Wolfram arrays before the call returns, so no result reads an arena a later call rewound.
- __Nothing costs anything until it is used.__
  A call reads the kernel's tensors in place, and an arena allocates only when a call writes into it.
- __Calls block.__
  Every verb returns once its call is final, on the kernel's own thread or an executor's workers, and there is one verb per operation rather than a twin per output style.
- __Handles are managed expressions.__
  The kernel frees an index, executor or arena once its last reference dies, and an integer a caller typed never reaches the library as a pointer.
  A call holds every handle it names until it returns, so `DenseSearch[DenseIndexOn[index, ExecutorForRam[]], queries, 10, ArenaInit[]]` is safe.
