# USearch for Wolfram

USearch provides a [Wolfram Mathematica](https://www.wolfram.com/mathematica/) integration via `LibraryLink`.
It exposes a compact set of functions to spawn an executor, construct an index, save and load checkpoints, add vectors, and search.

## Build

This directory is a subdirectory of the root project rather than a project of its own, so build it from the repository root.
Set `WOLFRAM_PATH` to your Mathematica installation if it is not detected automatically.

```sh
cmake -B build -D USEARCH_BUILD_LIB_WOLFRAM=1 -D WOLFRAM_PATH=/usr/local/Wolfram/Mathematica
cmake --build build --config Release --target USearchWFM

# Resulting artifact will be named like:
#   build/wolfram/USearchWFM.so     on Linux
#   build/wolfram/USearchWFM.dylib  on macOS
#   build/wolfram/USearchWFM.dll    on Windows
```

## Quickstart

Load the library and bind functions using `LibraryFunctionLoad`.
Below is a minimal example mirroring the API in `wolfram/test.wls`.

```wolfram
lib = If[$OperatingSystem === "Windows", "build\\wolfram\\USearchWFM.dll",
   If[$OperatingSystem === "MacOSX", "build/wolfram/USearchWFM.dylib", "build/wolfram/USearchWFM.so"]
];

ExecutorCreate  = LibraryFunctionLoad[lib, "ExecutorCreate",  {"UTF8String", Integer}, Integer];
ExecutorDrain   = LibraryFunctionLoad[lib, "ExecutorDrain",   {Integer}, "Void"];
ExecutorDestroy = LibraryFunctionLoad[lib, "ExecutorDestroy", {Integer}, "Void"];

DenseIndexCreate       = LibraryFunctionLoad[lib, "DenseIndexCreate",       {"UTF8String", "UTF8String", "UTF8String", Integer, Integer, Integer}, Integer];
DenseIndexDestroy      = LibraryFunctionLoad[lib, "DenseIndexDestroy",      {Integer}, "Void"];
DenseIndexSave         = LibraryFunctionLoad[lib, "DenseIndexSave",         {Integer, Integer, "UTF8String"}, "Void"];
DenseIndexLoad         = LibraryFunctionLoad[lib, "DenseIndexLoad",         {Integer, Integer, "UTF8String"}, "Void"];
DenseIndexSize         = LibraryFunctionLoad[lib, "DenseIndexSize",         {Integer}, Integer];
DenseIndexCapacity     = LibraryFunctionLoad[lib, "DenseIndexCapacity",     {Integer}, Integer];
DenseIndexDimensions   = LibraryFunctionLoad[lib, "DenseIndexDimensions",   {Integer}, Integer];
DenseIndexConnectivity = LibraryFunctionLoad[lib, "DenseIndexConnectivity", {Integer}, Integer];
DenseIndexAdd          = LibraryFunctionLoad[lib, "DenseIndexAdd",          {Integer, Integer, {Integer, 2}, {Real, _}}, "Void"];
DenseIndexSearch       = LibraryFunctionLoad[lib, "DenseIndexSearch",       {Integer, Integer, {Real, _}, {Integer, 3, "Shared"}, {Real, 2, "Shared"}, {Integer, 1, "Shared"}}, "Void"];

(* The sixteen identity bytes are the base-256 digits of the key, so an integer and a UUID
   reach the binding through one encoding, neither of them rounded *)
KeyBytes[key_Integer] := IntegerDigits[key, 256, 16];
KeyBytes[text_String] := IntegerDigits[FromDigits[StringDelete[text, "-"], 16], 256, 16];
KeyText[bytes_List] := StringInsert[StringJoin[IntegerString[bytes, 16, 2]], "-", {9, 13, 17, 21}];
KeyInteger[bytes_List] := FromDigits[bytes, 256];

(* Spawn the workers every fanned-out verb runs on: residency, threads *)
executor = ExecutorCreate["ram", 0];

(* Create an index: metric, quantization, residency, dimensions, capacity, connectivity *)
index = DenseIndexCreate["angular", "f32", "ram", 3, 10, 16];

(* Add a batch - one identity row per vector row, and a rank-1 vector is the one-row case *)
rows = {{0.2, 0.6, 0.4}, {0.4, 0.6, 0.2}};
DenseIndexAdd[executor, index, {KeyBytes["5f9c2b71-8e4d-4a63-b2f1-0c7a9d3e6b58"], KeyBytes[43]}, rows];

(* Search writes into three tensors you allocate, whose extents set the breadth *)
keys = ConstantArray[0, {2, 5, 16}];
distances = ConstantArray[0., {2, 5}];
counts = ConstantArray[0, 2];
DenseIndexSearch[executor, index, rows, keys, distances, counts];
Map[KeyText, keys, {2}]

(* Save, load, and cleanup *)
DenseIndexSave[executor, index, "index.safegraphs"];
DenseIndexLoad[executor, index, "index.safegraphs"];
ExecutorDrain[executor];
DenseIndexDestroy[index];
ExecutorDestroy[executor];
```

### Notes

- `ExecutorCreate` and `DenseIndexCreate` return opaque pointers encoded as integer handles, retired by `ExecutorDestroy` and `DenseIndexDestroy`.
- An executor owns the host threads or device streams; an index owns the rows.
  One executor serves several indexes, and its residency must match theirs.
- Every verb whose C counterpart takes an executor takes it as the leading argument.
- Vectors cross as real tensors, so they arrive as `f64` and the engine quantizes them into whatever the index stores.
  A rank-2 tensor is a batch of rows and a rank-1 tensor is a single row, so a whole batch crosses in one call rather than one crossing per vector.
  The engine checks the row width against the frozen geometry and refuses a mismatch rather than reading past the row.
- `DenseIndexSearch` writes into three tensors you allocate and pass as `"Shared"`: identities of `queries × wanted × 16`, distances of `queries × wanted`, and one match count per query.
  Three tensors rather than one, because sixteen identity bytes and a distance's mantissa cannot share a cell type without one of them being rounded, and a verb answers a single value.
  Their extents carry the breadth of the search, so it is no argument of its own, and rows past a query's match count are zeroed, since a truncated row is not an error.
  `ConstantArray` yields the packed arrays `"Shared"` passing needs, and the verb writes into them in place, so the symbols you handed it hold the answer once it returns.
- Keys are USearch's 128-bit identities, spanning sixteen integer cells apiece, most significant byte first.
  That is what `IntegerDigits[key, 256, 16]` renders and `FromDigits[cells, 256]` reads back, so an identity crosses at full precision whether it started as a machine integer, a UUID, or a number too wide for either.
  An integer identity is the encoding of a number below `2^64`, so it parks in the low eight bytes with the rest zeroed.
- Metrics are `inner`, `angular`, `euclidean`, `hamming`, and `jaccard`; quantizations are `f64`, `f32`, `bf16`, `f16`, `e5m2`, `e4m3`, `e3m2`, `e2m3`, `i8`, `i4x2`, `u8`, `u4x2`, and `u1x8`; residencies are `ram`, `vram`, and `nvme`.
- The candidate-pool breadth rides each call in the C interface rather than the handle, so it is no argument here and every verb derives it.
- Checkpoint paths carry the SafeGraphs extensions: `.safegraphs` for one monolith, `.safegraphs.json` for a manifest beside its shards.
