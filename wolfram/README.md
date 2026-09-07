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
DenseIndexAdd          = LibraryFunctionLoad[lib, "DenseIndexAdd",          {Integer, Integer, {Integer, 1}, {Real, _}}, "Void"];
DenseIndexSearch       = LibraryFunctionLoad[lib, "DenseIndexSearch",       {Integer, Integer, {Real, _}, Integer}, {Real, 3}];

(* Spawn the workers every fanned-out verb runs on: residency, threads *)
executor = ExecutorCreate["ram", 0];

(* Create an index: metric, quantization, residency, dimensions, capacity, connectivity *)
index = DenseIndexCreate["angular", "f32", "ram", 3, 10, 16];

(* Add a batch and search it - one row per key, and a rank-1 vector is the one-row case *)
rows = {{0.2, 0.6, 0.4}, {0.4, 0.6, 0.2}};
DenseIndexAdd[executor, index, {42, 43}, rows];

(* `queries * wanted * 2`, each innermost pair a key and its distance *)
matches = DenseIndexSearch[executor, index, rows, 5];

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
- `DenseIndexSearch` answers a `queries × wanted × 2` real tensor, each innermost pair a key and its distance.
  Pairs past a query's match count are zeroed, since a truncated row is not an error.
- Keys are machine integers, parked in the low eight bytes of USearch's 128-bit identity.
- Metrics are `inner`, `angular`, `euclidean`, `hamming`, and `jaccard`; quantizations are `f64`, `f32`, `bf16`, `f16`, `e5m2`, `e4m3`, `e3m2`, `e2m3`, `i8`, `i4x2`, `u8`, `u4x2`, and `u1x8`; residencies are `ram`, `vram`, and `nvme`.
- The candidate-pool breadth rides each call in the C interface rather than the handle, so it is no argument here and every verb derives it.
- Checkpoint paths carry the SafeGraphs extensions: `.safegraphs` for one monolith, `.safegraphs.json` for a manifest beside its shards.
