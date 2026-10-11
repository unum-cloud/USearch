# USearch for Wolfram

USearch provides a [Wolfram Mathematica](https://www.wolfram.com/mathematica/) integration via `LibraryLink`.
It exposes a compact set of functions to spawn an executor, construct an index, save and load checkpoints, insert, get and search vectors, and multiply a matrix against the index.
The `USearch.wl` package wraps the library, so every handle is a managed expression the kernel frees once its last reference dies.

## Build

This directory is a subdirectory of the root project rather than a project of its own, so build it from the repository root.
Set `WOLFRAM_PATH` to your Mathematica installation if it is not detected automatically.

```sh
cmake -B build_release -D USEARCH_BUILD_LIB_C=1 -D USEARCH_BUILD_LIB_WOLFRAM=1 -D USEARCH_DENSE_DTYPES=11 \
    -D WOLFRAM_PATH=/usr/local/Wolfram/Mathematica
cmake --build build_release --config Release --target usearch_wolfram

# Resulting artifact will be named like:
#   build_release/usearch_wolfram.so     on Linux
#   build_release/usearch_wolfram.dylib  on macOS
#   build_release/usearch_wolfram.dll    on Windows
```

`USEARCH_DENSE_DTYPES=11` compiles every dense storage type; the default four, `bf16`, `u1x8`, `i8` and `e4m3`, build faster but refuse the other seven.

It is the one library here without a `lib` prefix, because Wolfram finds a library dropped into `LibraryResources/$SystemID` by its bare name.

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

(* Search on the kernel's own thread, five matches per query, landing in an arena *)
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

Constructors address no index yet, so they take no receiver; the ones the C interface lets fan out take an `"Executor"` option instead.

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

## Executors

An executor owns the host threads or device streams; an index owns the rows.
One executor serves several indexes, and its residency must match theirs.
There is a constructor per C one, each taking every argument its own hardware has as an option, where zero and an empty list derive:

```wolfram
ExecutorForRam["MemoryDomains"->{0, 1}, "Threads"->0]
ExecutorOnPool[poolAddress, "MemoryDomains"->{}]                     (* borrows a fu_pool_t *)
ExecutorForVram["Devices"->{0}, "Streams"->{}, "ScratchLimit"->0]
ExecutorForNvme["Paths"->{"/dev/nvme0n1"}, "Threads"->0]
ExecutorDrain[executor, "parked"]                                    (* or "spinning" *)
```

`ExecutorDrain` picks whether the workers sleep or spin once idle.

## Residency and Serialization

Where the payload lives is the constructor's name, as in the C lifecycle verbs, so a residency an index lacks has no constructor to call.

```wolfram
DenseInitRam["inner", "bf16", 768, "MemoryDomains"->{0, 1}, "Connectivity"->0, "ValuesPerKey"->0]
DenseInitVram["inner", "bf16", 768, "Devices"->{0, 1, 2, 3}]
DenseLoadRam["index.safegraphs", "Executor"->executor, "MemoryDomains"->{}, "ValuesPerKey"->0]
DenseLoadVram["index.safegraphs", "Executor"->gpu, "Devices"->{0}]
DenseViewRam["index.safegraphs"]                                   (* read-only, served from the mapping *)
DenseViewNvme["index.safegraphs", "Executor"->drives, "Paths"->{"/dev/nvme0n1"}]
DenseMetadataRead["index.safegraphs"]                              (* geometry without loading a shard *)
```

Checkpoint paths carry the SafeGraphs extensions: `.safegraphs` for one monolith, `.safegraphs.json` for a manifest beside its shards.
A viewed index is read-only, and every mutating verb on it fails as `read-only`.
Until NVMe checkpoints carry keys, `DenseViewNvme` fails as `unsupported`.
In-memory buffers hold single monoliths only, as `UnsignedInteger8` numeric arrays:

```wolfram
monolith = DenseSaveBuffer[index, arena];
restored = DenseLoadRamBuffer[monolith];
DenseMetadataReadBuffer[monolith]
```

## Verbs

- `DenseReserve[receiver, capacity]` reserves the total capacity, including the current size, and `DenseClear[receiver]` erases every entry, keeping it.
- `DenseInsert[receiver, keys, vectors]` admits one row per key, and `DenseCount[receiver, key]` counts the entries under one.
- `DenseGet[receiver, key, arena]` answers every row stored under a key as a `rows × dimensions` real matrix, empty for an absent key.
- `DenseSearch[receiver, queries, wanted, arena]` answers `<|"Keys"->…, "Scores"->…|>`, one list per query, holding the matches it found, nearest first.
- `DenseMaxsim[receiver, queries, rowsPerQuery, wanted, arena]` scores multi-row queries late-interaction style, answering `<|"Keys"->…, "Scores"->…|>`.
- `DenseMultiply[receiver, left, arena]` runs on the `inner` metric and answers `<|"Offsets"->…, "Keys"->…, "Scores"->…, "Required"->…|>`: row `i` of the left matrix owns pairs `offsets[[i]] + 1` through `offsets[[i + 1]]`, in traversal order, and `"Wanted"->n` keeps at most that many pairs per row.
- `DenseRefine[receiver]` answers how many nodes had their neighbors changed, and `DenseHealth[receiver]` how many entries a search reaches.
- `DenseDescribe[receiver]` answers the metric, quantization, dimensions, connectivity, size, capacity, memory usage and dispatched capabilities.
- `VersionName[]`, `CapabilitiesCompiled[]` and `CapabilitiesAvailable[]` describe the library itself.

The candidate-pool breadth rides each call in the C interface rather than the handle, so it is an `"ExpansionFactor"` option on every verb that walks the graph, zero deriving it.
`DenseSearch` also takes `"Traversal"->"scan"`, which skips the walk and scores every row exactly, on RAM and NVMe handles.

## Notes

- Vectors cross as real tensors, so they arrive as `f64` and the engine quantizes them into whatever the index stores.
  A matrix is a batch of rows and a vector is a single row, so a whole batch crosses in one call rather than one crossing per vector.
  The tensor's own width crosses with the rows, and the engine refuses one that differs from the frozen geometry as `shape-mismatch`.
- Keys are USearch's 128-bit identities, written as integers below 2¹²⁸ and crossing as sixteen bytes, most significant first.
  That is what `IntegerDigits[key, 256, 16]` renders and `FromDigits[bytes, 256]` reads back, so an identity crosses at full precision whether it started as a machine integer, a UUID, or a number too wide for either.
  An integer identity is the encoding of a number below 2⁶⁴, so it parks in the low eight bytes with the rest zeroed.
  A UUID is the number its hexadecimal digits spell, `FromDigits[StringDelete[uuid, "-"], 16]`.
- A failing call answers `Failure[status, <|"MessageTemplate"->detail|>]`, where the status is the C name such as `shape-mismatch`, `io` or `read-only`, and the detail the engine's message.
- Metrics are `inner`, `angular`, `euclidean`, `hamming`, and `jaccard`; quantizations are `bf16`, `f16`, `e5m2`, `e4m3`, `e3m2`, `e2m3`, `i8`, `i4x2`, `u8`, `u4x2`, and `u1x8`; `DenseInitRam[dimensions]` makes an `angular` `bf16` index; moods are `parked` and `spinning`.

## Design

The package is a thin layer over the C ABI in `include/usearch/`, and every rule below follows from that.

- __Names follow the header.__
  Every function is its C verb without `usearch_` in the language's case, such as `DenseSearch`, `DenseInitRam` and `ExecutorForRam`; every option is a C field, and every usage message names the C symbol it calls or says it is binding-only.
- __Calls block.__
  Every verb returns once its call is final, on the kernel's own thread or an executor's workers, so a notebook never polls, waits or orders calls, and the C interface's deferred completion is left out.
  `DenseSearch[index, queries, 10, arena]` takes no executor, and `DenseIndexOn` only picks the workers.
- __Nothing costs anything until it is used.__
  A call reads the kernel's tensors in place, lands its outputs in the arena it is given, and allocates only the tensors it copies them into.
- __Every output whose length scales with the call lands in an arena.__
  Fixed-size reports such as descriptions, audits and counts stay plain values, and there is one verb per operation rather than a twin per output style.
- __Handles are managed expressions.__
  The kernel frees an index, executor or arena once its last reference dies, and an integer a caller typed never reaches the library as a pointer.
  A call naming a released handle by its ID fails as `uninitialized`.
- __Enums are the C tables' names.__
  Metrics, quantizations, moods and statuses cross as the strings `usearch_*_name` spells, read back through `usearch_*_from_name`, so the C library owns every spelling.
