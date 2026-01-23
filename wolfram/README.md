# USearch for Wolfram

USearch provides a [Wolfram Mathematica](https://www.wolfram.com/mathematica/) integration via `LibraryLink`.
It exposes a compact set of functions to construct, save/load/view, add vectors, and search.

## Build

Build the shared library using CMake. Set `WOLFRAM_PATH` to your Mathematica installation if it’s not detected automatically.

```sh
cd wolfram
# Optionally set the path to your Mathematica installation
# export WOLFRAM_PATH=/usr/local/Wolfram/Mathematica
cmake -B build
cmake --build build --config Release

# Resulting artifact will be named like:
#   build/libusearchWFM.so    on Linux
#   build/usearchWFM.dylib    on macOS
#   build/usearchWFM.dll      on Windows
```

## Quickstart

Load the library and bind functions using `LibraryFunctionLoad`.
Below is a minimal example mirroring the API in `wolfram/test.wls`.

```wolfram
lib = If[$OperatingSystem === "Windows", "build\\usearchWFM.dll",
   If[$OperatingSystem === "MacOSX", "build/usearchWFM.dylib", "build/libusearchWFM.so"]
];

IndexCreate      = LibraryFunctionLoad[lib, "IndexCreate",      {"UTF8String", "UTF8String", Integer, Integer, Integer, Integer, Integer}, Integer];
IndexSave        = LibraryFunctionLoad[lib, "IndexSave",        {Integer, "UTF8String"}, "Void"];
IndexLoad        = LibraryFunctionLoad[lib, "IndexLoad",        {Integer, "UTF8String"}, "Void"];
IndexView        = LibraryFunctionLoad[lib, "IndexView",        {Integer, "UTF8String"}, "Void"];
IndexDestroy     = LibraryFunctionLoad[lib, "IndexDestroy",     {Integer}, "Void"];
IndexSize        = LibraryFunctionLoad[lib, "IndexSize",        {Integer}, Integer];
IndexConnectivity= LibraryFunctionLoad[lib, "IndexConnectivity", {Integer}, Integer];
IndexDimensions  = LibraryFunctionLoad[lib, "IndexDimensions",  {Integer}, Integer];
IndexCapacity    = LibraryFunctionLoad[lib, "IndexCapacity",    {Integer}, Integer];
IndexAdd         = LibraryFunctionLoad[lib, "IndexAdd",         {Integer, Integer, {Real, 1}}, "Void"];
IndexSearch      = LibraryFunctionLoad[lib, "IndexSearch",      {Integer, {Real, 1}, Integer}, {Integer, 1}];

(* Create an index: metric, quantization, dimensions, capacity, connectivity, expansion_add, expansion_search *)
idx = IndexCreate["cos", "f32", 3, 10, 16, 64, 64];

(* Add a vector and search *)
vec = {0.2, 0.6, 0.4};
IndexAdd[idx, 42, vec];
keys = IndexSearch[idx, vec, 5];

(* Save, load/view, and cleanup *)
IndexSave[idx, "index.usearch"];
IndexLoad[idx, "index.usearch"]; (* or IndexView[idx, "index.usearch"] *)
IndexDestroy[idx];
```

### Notes

- `IndexCreate` returns an opaque pointer encoded as an integer handle.
- Vectors are passed as rank‑1 real tensors (`{Real, 1}`).
- Supported metrics and quantizations match the core USearch engine (e.g., `cos`, `ip`, `l2sq`; `f64`, `f32`, `f16`, `i8`, `b1x8`).
