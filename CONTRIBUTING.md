# Contribution Guide

Thank you for coming here!
It's always nice to have third-party contributors 🤗

---

To keep the quality of the code high, we have a set of guidelines common to [all Unum projects](https://github.com/unum-cloud).

- [What's the procedure?](https://github.com/unum-cloud/awesome/blob/main/Workflow.md#organizing-software-development)
- [How to organize branches?](https://github.com/unum-cloud/awesome/blob/main/Workflow.md#branches)
- [How to style commits?](https://github.com/unum-cloud/awesome/blob/main/Workflow.md#commits)

## Before you start

Before building the first time, please pull `git` submodules.
That's how we bring in NumKong and other optional dependencies to test all of the available functionality.

```sh
git submodule update --init --recursive
```

The repository carries its own hooks, gating each staged file against the formatter its language uses and skipping the rest:

```sh
git config core.hooksPath .githooks
```

## C++ 20 & C 99

Our primary C++ implementation uses CMake for builds.
The headers need C++20, the C interface in `c/usearch/*.h` stays C99, and the Python and JavaScript glue around it compiles as C11.
If this is your first experience with CMake, use the following commands to get started:

```sh
sudo apt-get update && sudo apt-get install cmake build-essential libjemalloc-dev # Ubuntu
brew install llvm cmake # macOS
```

Using modern syntax, this is how you build and run the test suite:

```sh
cmake -D USEARCH_BUILD_TEST=1 -D CMAKE_BUILD_TYPE=Debug -B build_debug
cmake --build build_debug --config Debug
build_debug/usearch_test
```

If the build mode is not specified, the default is `Release`.

```sh
cmake -D USEARCH_BUILD_TEST=1 -B build_release
cmake --build build_release --config Release
build_release/usearch_test
```

For development purposes, you may want to include symbols information in the build:

```sh
cmake -D USEARCH_BUILD_TEST=1 -D CMAKE_BUILD_TYPE=RelWithDebInfo -B build_relwithdebinfo
cmake --build build_relwithdebinfo --config RelWithDebInfo
build_relwithdebinfo/usearch_test
```

`usearch_test` and `usearch_test_c` open by naming the build and the machine it runs on:

```text
USearch 3.0.0
- Compiled for: serial,neon,neonsdot,neonbfdot,neonfhm,neonfp8
- This machine: serial,neon,neonsdot,neonbfdot,neonfhm
```

Both lists come from the C interface, the same strings the bindings report.
"Compiled for" is `usearch_capabilities_compiled()`, the NumKong capability levels this build carries kernels for, and "This machine" is `usearch_capabilities_available()`, those of them this processor can also run.
`usearch_test_vram` runs on the GPUs `USEARCH_DEVICES` names, comma-separated USearch device ordinals like `0` or `0,2`, positions in the listing `usearch_gpu_devices` reports: NVIDIA devices first, then AMD, then Apple.
Unset, it takes the first device of every vendor present, and an ordinal past the listing prints the listing and exits 1.
It links one object library per vendor the build carries, and each device opens with its vendor's line: `- CUDA: <device> nvidia-<class> #<ordinal>, <N> SMs, <M> MiB`, `- ROCm: device #<ordinal> amd-cdna<N>` or `- Metal: device #<ordinal> apple-metal<N>`.
Without any GPU it prints `- GPU: none listed` and exits 0, skipping every GPU test.
`USEARCH_NVME_PATHS` names NVMe namespaces the same way, comma-separated in the grammar of `usearch_nvme_options_t::paths`: files, block devices or SPDK transport IDs like `trtype:PCIe traddr:0000:01:00.0`.
Unset, the SPDK suite saves `usearch-nvme-spdk.safegraphs` under `TMPDIR` for a rig to copy onto a namespace, and the next run, naming that namespace, views the copy.
`usearch_test_c` reads `USEARCH_SEED` and `USEARCH_FILTER` too, the filter as a substring of its test names.

Those three configurations, and the benchmarking one from [bench/README.md](bench/README.md), are also packaged as CMake presets, so the flag soup below is optional:

```sh
cmake --list-presets              # the ones this host can build, from `debug` to `wasm32_wasi` and `msvc2026_x64`
cmake --preset release            # configures into `build_release`
cmake --build --preset release    # builds it
ctest --preset release            # runs the suites
```

Every preset writes into `build_<preset name>` and exports a `compile_commands.json` for the language server.
The development ones build with LLVM through Ninja and link with `lld`; `tsan` uses GCC, and CI rows override the compiler with `-D CMAKE_CXX_COMPILER`.
`check` compiles every suite at `-O0` with warnings as errors.
The rest match CI: `library` for the C library alone, `linux_arm64`, `android_arm64`, `android_armv7`, the three WASM toolchains, and `msvc2026_x64` / `msvc2022_arm64`.
Three of them are about GPUs: `cuda` compiles the CUDA suites for the host GPU alone, which is the fastest way to iterate and the tree the repo's `.clangd` reads, while `cuda_release` emits a fat binary spanning every architecture through both Blackwell lines; `rocm` and `metal` do the same for AMD and Apple.
`xeon4_hopper`, `xeon6_blackwell`, `grace_hopper` and `vera_rubin` compile the C library for one machine alone: its CPU through `USEARCH_MARCH` in a monoversioned build, `USEARCH_MULTIVERSIONED=OFF`, and its one GPU code through `USEARCH_CUDA_ARCH`.
`cuda_trace` adds `-ftime-trace`, dropping a profile beside every object for when a translation unit starts dominating the build.
Machine-specific overrides - a different compiler, `ccache`, a CUDA toolkit outside the default prefix - belong in an untracked `CMakeUserPresets.json` beside it:

```json
{
    "version": 6,
    "configurePresets": [
        {
            "name": "cuda_local",
            "inherits": "cuda",
            "binaryDir": "${sourceDir}/build_cuda",
            "cacheVariables": {
                "CMAKE_CUDA_HOST_COMPILER": "/usr/bin/g++-14",
                "CMAKE_CXX_COMPILER_LAUNCHER": "ccache"
            }
        }
    ]
}
```

The CUDA presets need CMake 3.31 or newer - the release that learned the Blackwell architectures - while the CPU-only build stays on the 3.28 floor.
Ubuntu's own package predates the former, so [Kitware's repository](https://apt.kitware.com) is the way to get it.

The CMakeLists.txt file has a number of options you can pass:

- What to build:
  - `USEARCH_BUILD_TEST` - build the C++ test suite, and the C one alongside `USEARCH_BUILD_SHARED`
  - `USEARCH_BUILD_BENCH` - build the C++ benchmark suite
  - `USEARCH_BUILD_SHARED` - build and install the shared and static C libraries by default; without it they build only for a target that links them
  - `USEARCH_BUILD_STUB` - a brute-force stand-in answering the same verbs over the standard library alone, for isolating a binding from the engine while bisecting one.
    It carries the same library names in a `stub/` subdirectory, so pointing a binding at it is a search path rather than a different link line.
  - `USEARCH_CUDA_ARCH`, `USEARCH_ROCM_ARCH`, `USEARCH_METAL_ARCH` - build the NVIDIA, AMD or Apple variants and suites for the codes they name below, each empty and off unless a preset or `-D` names one
  - `USEARCH_DENSE_DTYPES` - how many dense storage dtypes the library, tests and benchmarks compile, from the front of bf16, u1, i8, e4m3, f16, u8, e5m2, i4, u4, e3m2, e2m3, `4` by default and `11` wherever a prebuilt package comes from: the `library`, `wasm` and `msvc2026_x64` presets, `pyproject.toml`, the npm scripts, `build.gradle`, `pixi.toml` and the per-binding recipes below. Binding suites skip the kinds a narrower build refuses.
    The C interface refuses a dtype the build left out with `usearch_status_unsupported_k`; the Rust crate builds from source and keeps the default four.
- Which CPUs to target:
  - `USEARCH_MARCH` - the floor of USearch's own units, seeded from the environment variable of the same name: empty for the architecture's floor, an ISA level like `x86-64-v3`, `native` for this host, or a CPU name like `graniterapids` or `neoverse-v2`, also when cross-compiling.
    Every class unit adds its class's flags over it, the tests and benchmarks drop their host tuning for it, and NumKong, StringZilla, ForkUnion and SmashTable take the same value.
  - `USEARCH_MULTIVERSIONED` - the build kind, `ON` by default and seeded from the environment too: a multiversioned build compiles every CPU class the toolchain compiles above the floor, picked at runtime, and with `OFF` a monoversioned build compiles one `monoversioned` class and no named class, `serial` included.
    The monoversioned class carries every capability NumKong, StringZilla and ForkUnion enabled: what the floor implies, plus what `-D <LIB>_TARGET_<CAP>=1` vouches for where no compiler macro reports it, like `FORKUNION_TARGET_ARM64_WFET=1` on Vera.
    Each modality and residency compiles one unit for it, and a handle pinned below it, or run on a CPU lacking it, gets `usearch_status_unsupported_k`.
    `-D USEARCH_MARCH=native -D USEARCH_MULTIVERSIONED=OFF` is the native build, and the siblings take the same build kind.
- Which GPUs to target:
  - `USEARCH_CUDA_ARCH` - the NVIDIA code the library, tests and benchmarks compile for: `all` for the release set, `native` for each GPU `nvidia-smi` lists, mapped to its family code like `100f` on a B200 or B300, or one code like `90a`.
    A GPU class exists only when a code names it, `100`, `100a`, `100f`, `103a` or `103f` for datacenter Blackwell, so `100f` compiles the Blackwell kernels and the vendor-wide `nvidia` unit alone.
    The release set compiles each class at the lowest code its kernels need, as PTX alone where a plain code suffices, so every later GPU runs it, and never below CUDA 13's Turing floor: `compute_80` for the vendor-wide unit, whose sequence, shapes, geospatial and basic kernels carry NumKong's Ampere `bf16x2` casts, and for Ampere's `mma.sync`, `compute_89` for Ada's FP8, `compute_90` for Hopper, which issues no `wgmma`, and `compute_120` for the RTX line.
    Datacenter Blackwell's `tcgen05` needs a family code, so its class carries the `100f` cubin instead.
    A native or named code compiles as SASS alone, with no PTX.
    NVCC compiles only device code, into fatbins the binaries embed; nothing links `libcuda` or the CUDA runtime, as the library opens the driver at first use.
    A NumKong GPU tier turns on in the CUDA tests only when every code runs it, so the release set exercises the Ampere one; iterate on one GPU with its family code, like `100f`, to test that generation's kits.
  - `USEARCH_ROCM_ARCH` - the AMD processor: `all` for gfx942, gfx950, gfx1250 and gfx1251, `native` for those `rocm_agent_enumerator` lists, or one like `gfx942`; each class embeds a code object per resolved processor it runs on, and a class with none does not exist.
  - `USEARCH_METAL_ARCH` - the Apple family: `all` for the `apple7`, `apple9` and `apple10` classes, or one of them, as Metal compiles the kernels on the device.
- Which other dependencies to use:
  - `USEARCH_WITH_JEMALLOC` - use Jemalloc for memory management
- How strict to be:
  - `USEARCH_WERROR` - treat compiler warnings as errors, as the `check` preset does

The sibling libraries carry no switches.
NumKong, ForkUnion, StringZilla and SmashTable are what the headers are written against, so every build acquires all four - from the submodule when a recursive clone provides one, and from the pinned commit otherwise.
The language bindings take their headers from `numkong/include`, `forkunion/include`, and `stringzilla/include` in the checkout, so `pyproject.toml`, `package.json`, `build.rs`, and `Package.swift` each reach the same roots, and none of them builds from a non-recursive clone.

Putting all of this together, compiling all targets on most platforms should work with the following snippet:

```sh
cmake -D CMAKE_BUILD_TYPE=Release -D USEARCH_WITH_JEMALLOC=1 -D USEARCH_BUILD_TEST=1 -D USEARCH_BUILD_BENCH=1 -D USEARCH_BUILD_SHARED=1 -B build_release

cmake --build build_release --config Release
build_release/usearch_test
build_release/usearch_test_c
```

Similarly, to use the most recent Clang compiler version from Homebrew on macOS:

```sh
brew install llvm cmake
cmake \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_C_COMPILER="$(brew --prefix llvm)/bin/clang" \
    -D CMAKE_CXX_COMPILER="$(brew --prefix llvm)/bin/clang++" \
    -D USEARCH_WITH_JEMALLOC=1 \
    -D USEARCH_BUILD_TEST=1 \
    -D USEARCH_BUILD_BENCH=1 \
    -D USEARCH_BUILD_SHARED=1 \
    -B build_release

cmake --build build_release --config Release
build_release/usearch_test
build_release/usearch_test_c
```

Linting:

```sh
cppcheck --enable=all --force --suppress=cstyleCast --suppress=unusedFunction \
    include/usearch/utilities/types.hpp \
    include/usearch/basic/ram_index.hpp \
    include/usearch/dense/ram_index.hpp \
    include/usearch/dense/ram_metric.hpp
```

I'd recommend putting the following breakpoints when debugging the code in GDB:

- `__asan::ReportGenericError` - to detect illegal memory accesses.
- `__ubsan::ScopedReport::~ScopedReport` - to catch undefined behavior.
- `__GI_exit` - to stop at exit points - the end of running any executable.
- `__builtin_unreachable` - to catch all the places where the code is expected to be unreachable.

### Capacities & Bounds

No cache, pool, selection, batch or buffer takes its size from a constant picked to fit, like `64` or `2 * connectivity`.
Each size follows from what its consumer holds, such as a walk's whole pool, or a row's live neighbors, one row of newcomers and its own row, or it comes from the caller.
Derive each bound once on the host and carry it to the kernels in the launch record, so that no kernel and no vendor tower recomputes it with a formula of its own.
Hardware properties, like core counts, SIMD widths and shared-memory sizes, are read from the device at run time and never written into the code.
Nothing is dropped past a capacity: excess candidates fold in rounds, and a failed allocation is counted.
Graph walks have no hop budgets, thresholds or upper bounds that end them early.
A walk runs until it converges, and a fixed device structure that fills up grows or falls back to a counted re-score.

A batch width is written once, where the batch is born.
A quad is `quad_lanes_k` lanes, the candidates one NumKong finalize scores together, and `quad<T>` in `include/usearch/utilities/types.hpp` spells one.
A producer dispatches a round's shorter tail to its own static extent once, through `with_static_extent`, so callbacks only ever see static extents.
A consumer sizes its buffers from the span it is handed, as `std::array<admission_t, extent_>`, and never names the width again.
A width restated in a second place drifts from the first the day either one changes.

A hot verb allocates nothing once warm.
Its traversal contexts come from the executor's per-thread arenas, which keep their widest chunk between calls, so a call only bumps pointers, and only a call wider than every earlier one grows them.
A buffer every step writes before reading is carved without a fill, a resize within capacity keeps its memory, and `reset` is the one verb that releases it.

### No Tuning Knobs

Every behavior follows from the inputs, the index's own geometry and the hardware it runs on, never from a hyper-parameter someone tuned.
Don't add a flag, a constant or a heuristic whose value was searched until a benchmark looked good: such values fit one dataset and one machine, and they multiply into combinations nobody can test.
When a choice has no derivation yet, derive it before landing it, from a cost model, a hardware property or a bound the algorithm guarantees, and state that derivation where the value is defined.
A value that can only be measured, like an empirical wave width, says so in its docstring, names what was measured, and lives in one definition every backend reads.

### Numeric Precision

Every score, distance and accumulator type comes from NumKong's per-input result types in `numkong/types.hpp`, read off the stored scalar or the kit wrapping it.
The dense tier stores `bf16`, `u1`, `i8`, `e4m3`, `f16`, `u8`, `e5m2`, `i4`, `u4`, `e3m2` and `e2m3`, whose inner products answer in `f32`, `i32` or `u32` through `dot_result_t`.
A sum over many of those products follows NumKong's reductions through `reduce_moments_sum_t`: `f32` for the floating formats, saturating `i64` and `u64` for the integers.
No dense computation needs `f64`, and widening to the largest type to be safe moves twice the bytes through every hot buffer.
The C interface's `double` `usearch_distance_t` is where scores land, never what they are computed in.

### Hot Entries

Every struct a thread keeps or carries along a hot path is flat and 8, 16 or 32 bytes, like `scored_node_t`, `nvme_flight` and `nvme_walk_payload`.
Lay its fields out directly rather than nesting structs, whose padding compounds, and pin its size with a `static_assert` beside the definition.
A would-be 24-byte entry shrinks first, through 32-bit node IDs or the kernel's own result type, and otherwise grows to 32 by a field that earns its place.

### Predicates

A predicate carries its cost, `usearch_predicate_cost_t` in `c/usearch/types.h`, and the engines ask it accordingly.
A cheap one is asked as candidates arrive, a full quad or a round's shorter tail at a static extent, with nothing buffered.
An expensive one gathers each thread's candidates beside the engine's payloads and answers them in one call per traversal round or scan run.
Those per-thread buffers are the only buffers a predicate ever holds, and an engine opens a per-thread family and reserves its buffers in the executor's arenas right after the call's rewind.

### Planned Operations

A long or multi-phase operation starts as a plan, like `plan_refine` or `plan_insert`, and its blocking verb is only a loop driving that plan.
`steps()` lists the current phase's steps, `run(step, context)` runs any of them on the calling thread, `advance()` is the barrier answering whether another phase follows, and `finish()` publishes and releases.
Planning claims and reserves everything a step writes, so a step never fails for lack of memory and a plan stopped between steps leaves the index correct.
Per-thread state lives in the traversal context `run` is handed, in buffers it already holds where they fit, as attention keeps its softmax states past the queries in `staged_queries`.
An operation is a plan only when its work crosses a barrier between phases, runs long enough to stop and resume, or writes the index; a pass whose threads each finish their own share, like `search`, `multiply`, `attend` or MaxSim, is a direct verb over the caller's contexts.
A plan owns only what crosses its barriers: scratch belongs to the context, results land in the caller's outputs, named at planning, and whatever follows from other members is computed rather than stored.
`run`, `advance` and `finish` never allocate.

### Components

Logic is never shared through inheritance: no base classes, CRTP or mixins.
Each residency and each GPU vendor is a dedicated tower with its own customizable logic.
Code two towers genuinely share becomes a component each tower owns as a member, or a free function each tower calls.
Host and device towers never fuse, since a RAM walk is a much finer-grained processing mode than a device batch.
A tower never takes a lock to make a shared design fit; it redesigns the protocol instead.

### CPU Classes

Engines never reach for the standard library's waits or yields, and never hard-code a waiter.
Each template in `include/` takes its kits, cache hints, waiter and atomic reference as template arguments, and `src/` picks them per CPU class, as [Dispatch over Capability Kits](src/README.md#dispatch-over-capability-kits) describes.
An engine that needs richer behavior, like prefetching whole objects, wraps the primitive it was handed.
No template names a CPU class, an ISA instruction or a topology query, and capability masks stay in `src/`, where rungs are chosen.

### Spelling Types

New code never names or sizes a type through `decltype`, `decltype(auto)` included.
Spell the type, or export it as a member alias where it is defined and read it from there, as `typename codec_t::view_t` names what a codec's `view` returns.
A `decltype` makes every reader evaluate an expression to learn a type its author already knew.
Templates carry no `_t` or `_k` suffix, whether class, alias or variable templates or concepts, while concrete types end in `_t` and constants in `_k`.

### Cross Compilation

Unlike GCC, LLVM handles cross compilation very easily.
You just need to pass the right `TARGET_ARCH` and `BUILD_ARCH` to CMake.
The [list includes](https://packages.ubuntu.com/search?keywords=crossbuild-essential&searchon=names):

- `crossbuild-essential-amd64` for 64-bit x86
- `crossbuild-essential-arm64` for 64-bit Arm
- `crossbuild-essential-armhf` for 32-bit ARM hard-float
- `crossbuild-essential-armel` for 32-bit ARM soft-float (emulates `float`)
- `crossbuild-essential-riscv64` for RISC-V
- `crossbuild-essential-powerpc` for PowerPC
- `crossbuild-essential-s390x` for IBM Z
- `crossbuild-essential-mips` for MIPS
- `crossbuild-essential-ppc64el` for PowerPC 64-bit little-endian

Here is an example for cross-compiling for Arm64 on an x86_64 machine:

```sh
sudo apt-get update
sudo apt-get install -y clang lld make crossbuild-essential-arm64 crossbuild-essential-armhf
export CC="clang"
export CXX="clang++"
export AR="llvm-ar"
export NM="llvm-nm"
export RANLIB="llvm-ranlib"
export TARGET_ARCH="aarch64-linux-gnu" # Or "x86_64-linux-gnu"
export BUILD_ARCH="arm64" # Or "amd64"

cmake -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_C_COMPILER_TARGET=${TARGET_ARCH} \
    -D CMAKE_CXX_COMPILER_TARGET=${TARGET_ARCH} \
    -D CMAKE_SYSTEM_NAME=Linux \
    -D CMAKE_SYSTEM_PROCESSOR=${BUILD_ARCH} \
    -B build_artifacts
cmake --build build_artifacts --config Release
```

The `android_arm64` and `android_armv7` presets need Android NDK r28 or newer, whose libc++ ships `std::atomic_ref`, found through `ANDROID_NDK_ROOT`.
For Android development, you can also cross-compile for ARM architectures without requiring the full Android NDK setup.
Here's an example targeting 32-bit ARM (`armeabi-v7a`):

```sh
sudo apt-get update
sudo apt-get install -y clang lld crossbuild-essential-armhf

# Cross-compile for 32-bit ARM (Android compatible)
CMAKE_TRY_COMPILE_TARGET_TYPE=STATIC_LIBRARY \
cmake -B build_artifacts \
  -D CMAKE_C_COMPILER=clang \
  -D CMAKE_CXX_COMPILER=clang++ \
  -D CMAKE_C_COMPILER_TARGET=armv7-linux-gnueabihf \
  -D CMAKE_CXX_COMPILER_TARGET=armv7-linux-gnueabihf \
  -D CMAKE_SYSTEM_NAME=Linux \
  -D CMAKE_SYSTEM_PROCESSOR=armv7 \
  -D CMAKE_C_FLAGS="--target=armv7-linux-gnueabihf -march=armv7-a" \
  -D CMAKE_CXX_FLAGS="--target=armv7-linux-gnueabihf -march=armv7-a" \
  -D CMAKE_BUILD_TYPE=RelWithDebInfo \
  -D USEARCH_BUILD_SHARED=1

cmake --build build_artifacts --config RelWithDebInfo
file build_artifacts/libusearch_shared.so # Verify the output
# Should show: ELF 32-bit LSB shared object, ARM, EABI5 version 1 (SYSV), dynamically linked, ...
```

The resulting `libusearch_shared.so` can be used in Android projects by placing it in `src/main/jniLibs/armeabi-v7a/` for 32-bit ARM or `arm64-v8a/` for 64-bit ARM.

## Python 3

Python bindings are a CPython extension available on [PyPI](https://pypi.org/project/usearch/).
`scikit-build-core` drives the project's own CMake from `pyproject.toml`, so the extension is a CMake target rather than a second native build described in a `setup.py`.
To install USearch locally using `uv`:

```sh
uv venv --python 3.12                   # the oldest version `pyproject.toml` admits
source .venv/bin/activate               # to activate the virtual environment
uv pip install -e . --force-reinstall   # to build locally from source
```

Or using `pip` directly:

```sh
pip install -e . --force-reinstall
```

For testing USearch uses PyTest, which is pre-configured in `pyproject.toml`.
The CI runs it with `-x`, stopping at the first failure.

```sh
uv pip install -e . --group test              # the suite and everything it imports
python -X faulthandler -m pytest -x           # how the CI runs it
```

The `seed` fixture hands a test 42, unless `USEARCH_SEED` names another seed or is `random`, which draws a fresh one per run.
pytest prints every setting in its header, as `- Seed: <n>`, so a failure under a drawn seed reruns with that number.
Under pytest-repeat, the fixture adds the repeat step to the seed, so repetitions differ and each one still reproduces.
`USEARCH_DIMS` replaces the swept dimension counts, and `USEARCH_FILTER` keeps the tests whose node ID matches it, as a regex or else as a substring.
A value that does not parse stops the run with `USEARCH_SEED="<value>" does not parse, expected an unsigned integer or random`.

```sh
USEARCH_SEED=random python -m pytest test/               # draws a seed and prints it in the header
USEARCH_SEED=1528003704 python -m pytest test/ --count 3 # replays that seed, then the next two
USEARCH_FILTER=quantization python -m pytest test/       # only the tests whose node id mentions it
```

Linting:

```sh
uv pip install -e . --group lint
ruff check --output-format=github python
```

Before merging your changes you may want to test your changes against the entire matrix of Python versions USearch supports.
For that you need the `cibuildwheel`, which is tricky to use on macOS and Windows, as it would target just the local environment.
Still, if you have Docker running on any desktop OS, you can use it to build and test the Python bindings for all Python versions for Linux:

```sh
pip install cibuildwheel
cibuildwheel
cibuildwheel --platform linux                   # works on any OS and builds all Linux backends
cibuildwheel --platform linux --archs x86_64    # 64-bit x86, the most common on desktop and servers
cibuildwheel --platform linux --archs aarch64   # 64-bit Arm for mobile devices, Apple M-series, and AWS Graviton
cibuildwheel --platform macos                   # works only on macOS
cibuildwheel --platform windows                 # works only on Windows
```

You may need root privileges for multi-architecture builds:

```sh
sudo $(which cibuildwheel) --platform linux
```

On Windows and macOS, to avoid frequent path resolution issues, you may want to use:

```sh
python -m cibuildwheel --platform windows
```

## JavaScript

USearch provides N-API bindings for NodeJS available on [NPM](https://www.npmjs.com/package/usearch).
`cmake-js` drives the project's own CMake from `package.json`, so the addon is a CMake target rather than a second native build described in a `binding.gyp`.
If you don't have NPM installed, first the Node Version Manager:

```sh
wget -qO- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
nvm install 22
```

Testing:

```sh
npm install --ignore-scripts    # the dependencies alone
npm run prebuild                # `build-native` compiles `javascript/lib.c` and the engine, then stages them under `prebuilds/`
npm run build-js            # Generate JS from TS
npm test                    # Run the test suite
```

To compile for AWS Lambda you'd need to recompile the binding.
You can test the setup locally in a Docker image:

```Dockerfile
FROM public.ecr.aws/lambda/nodejs:22-x86_64
RUN npm init -y
# Amazon Linux 2023 packages CMake 3.22, older than the 3.28 floor, so pip brings a newer one.
RUN microdnf install -y tar git gcc-c++ make python3-pip && pip3 install "cmake>=3.28"
RUN npm install --build-from-source usearch
```

To compile to WebAssembly make sure you have `emscripten` installed and pick one of the checked-in toolchains.
There is no WebAssembly target of its own — the C library is what crosses, and a module carries one SIMD tier, width and threads choice, so each toolchain file fixes all three:

- `cmake/toolchain-wasm32-emscripten.cmake` builds the `wasm32_v128_k` class — 32-bit, single-threaded, the fixed-width `simd128` tier every engine loads.
- `cmake/toolchain-wasm64-emscripten.cmake` builds the `wasm64_extended_k` class — relaxed SIMD, shared memory for real thread pools, and 64-bit addressing.
- `cmake/toolchain-wasm32-wasi.cmake` builds `wasm32_v128_k` against the WASI SDK, for Wasmtime, Wasmer and the Rust crate.

```sh
emcmake cmake -B build_wasm32_emscripten -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-wasm32-emscripten.cmake -DUSEARCH_BUILD_SHARED=1 -DUSEARCH_BUILD_TEST=1
cmake --build build_wasm32_emscripten --target usearch_static usearch_test_c
ctest --test-dir build_wasm32_emscripten -R '^usearch_test_c$'
```

The toolchain file must be passed as the single argument `-DCMAKE_TOOLCHAIN_FILE=…`; written with a space, `emcmake` appends its own file, which then wins.
If you don't yet have `emcmake` installed:

```sh
git clone https://github.com/emscripten-core/emsdk.git && ./emsdk/emsdk install latest && ./emsdk/emsdk activate latest && source ./emsdk/emsdk_env.sh
```

## Rust

USearch provides Rust bindings available on [Crates.io](https://crates.io/crates/usearch).
`build.rs` compiles no C or C++ of its own — it drives the project's CMake and links what it produced.

Naming no feature links the engine, and `cuda` links it with the CUDA backends compiled in.
Each feature is a whole implementation of the C interface rather than a knob on one, so `build.rs` refuses a pair — never `--all-features`.

```sh
cargo test -- --nocapture --test-threads=1
cargo clippy --all-targets -- -D warnings   # how the CI lints
cargo fmt --check
```

Publishing the crate is a bit more complicated than normally.
If you simply pull the repository with submodules and run the following command it will list fewer files than expected:

```sh
cargo package --list --allow-dirty
```

The reason for that is the heuristic that Cargo uses to determine the files to include in the package.

> Regardless of whether exclude or include is specified, the following files are always excluded:
> Any sub-packages will be skipped (any subdirectory that contains a Cargo.toml file).

Assuming both NumKong and StringZilla contain their own `Cargo.toml` files, we need to temporarily exclude them from the package.

```sh
mv numkong/Cargo.toml numkong/Cargo.toml.bak
mv stringzilla/Cargo.toml stringzilla/Cargo.toml.bak
cargo package --list --allow-dirty
cargo publish

# Revert back
mv numkong/Cargo.toml.bak numkong/Cargo.toml
mv stringzilla/Cargo.toml.bak stringzilla/Cargo.toml
```

## Zig

USearch provides Zig bindings under `zig/`, driven by `build.zig` and needing Zig 0.16 or newer.
No C or C++ is compiled there: the engine is CMake's to build, and `-Dlib-dir` names a directory holding it, repeatable.
It takes the shared library rather than the archive, since Zig answers `linkSystemLibrary("stdc++")` with its own libc++, which leaves a GCC-built archive short of symbols.

```sh
cmake -B build_release -D USEARCH_DENSE_DTYPES=11
cmake --build build_release --target usearch_shared

zig build                                     # compile check, resolving no symbols
zig build test -Dlib-dir=build_release        # links, so every `extern fn` must resolve
zig fmt --check zig build.zig
```

`build.zig.zon` pins its version against the tracked `VERSION` file at comptime, so bumping one and forgetting the other fails the build.

## Mojo

The Mojo binding lives under `mojo/` and `dlopen`s the engine rather than linking it, so a consumer needs no linker flags of its own.
Pixi drives both halves: the root `pixi.toml` declares the `usearch` package the tree's own CMake builds, and `mojo/pixi.toml` the Mojo package depending on it.
The engine lands in the environment prefix, which is the runpath `mojo build` bakes.

```sh
pixi run test
pixi run build
pixi run mojo format -l 120 mojo/test.mojo
```

Mojo 1.0 has no test discovery, so every check is registered in `main` by hand and one that is not silently never runs.

## Swift

USearch provides Swift bindings through the [Swift Package Manager](https://swiftpackageindex.com/unum-cloud/USearch).
Swift 6.4 or newer is required: the streaming and checkpoint verbs take `MutableSpan` and `RawSpan`, and `Key` stores its bytes in an `InlineArray`.

SwiftPM links the C library prebuilt, as the `USearchC` XCFramework on Apple platforms and as an SE-0482 artifact bundle elsewhere, which the `swift` preset builds for the host.
`USEARCH_SWIFT_ARTIFACT` points `Package.swift` at it, relative to the package root, and `NUMKONG_SWIFT_ARTIFACT` points the `numkong/` checkout at the one its own `swift` preset builds; without them, SwiftPM downloads the releases':

```sh
(cd numkong && cmake --preset swift && cmake --build --preset swift)
cmake --preset swift
cmake --build --preset swift

export USEARCH_SWIFT_ARTIFACT=build_swift/USearchC.xcframework # USearchC.artifactbundle off Apple platforms
export NUMKONG_SWIFT_ARTIFACT=build_swift/NumKongC.xcframework # NumKongC.artifactbundle off Apple platforms
swift build
swift test
```

The suite is Swift Testing: tests are `@Test` functions grouped into `@Suite` types, and assertions read as `#expect(...)`.
It covers what the binding itself decides — key byte order, options marshalling through the residency union, matrix lowering, handle lifetime and the span borrows — since that vector search returns near neighbors is the engine's contract and is tested where the engine lives.

Formatting is enforced with Apple's `swift-format`, whose style lives in the `.swift-format` file at the repository root:

```sh
swift format --in-place --recursive swift Package.swift
swift format lint --strict --recursive swift Package.swift
```

Neither the build nor the suite needs an Apple device — both run on Linux, where [`swift.org/install`](https://www.swift.org/install) carries the toolchains.
A toolchain built for an older Ubuntu than the host may want that release's `libxml2` and `libicu` alongside it, since it links their older sonames.

## Go

USearch provides Go bindings, that depend on the C library that must be installed beforehand.
So one should first compile the C library, link it with Go, and only then run tests.

```sh
cmake -B build_release -D USEARCH_BUILD_SHARED=1 -D USEARCH_DENSE_DTYPES=11
cmake --build build_release --config Release -j

cp build_release/libusearch_static.a golang/  # the archive `golang/types.go` links, NumKong and ForkUnion included
cp -r c/usearch golang/                              # to make the headers available to Go

cd golang && go test -race -v ./... ; cd ..
```

The race detector matters here: the tests drive one shared handle from many goroutines, and it is the only thing that catches a missing `runtime.KeepAlive` around a matrix crossing into C.

Compilation and the static checks need only the headers, not the library, so they run before it links:

```sh
cp -r c/usearch golang/

cd golang
go build ./...      # compiles the cgo preamble and shim.c against the real headers
go vet ./...        # catches the cgo pointer-rule violations the headers cannot
gofmt -l .          # must print nothing
staticcheck ./...   # if installed
golangci-lint run   # if installed
```

## Java, Kotlin & Scala

The Java SDK reaches the C interface through the FFM API, so it compiles no native boundary of its own — there is no JNI shim and no `usearch_java` target.
Java 25 is the floor, and the runtime needs `--enable-native-access`, which `build.gradle` already passes to every test task.

Gradle drives CMake for the engine, so a plain build needs nothing else:

```sh
gradle clean build --warning-mode=all   # build, engine included
gradle test --rerun-tasks               # the Java suite
gradle spotlessApply                    # format Java, Kotlin and Scala
```

`-PusearchPrebuilt=<directory>` points every suite at a library that is already built and skips the CMake tasks, which is how CI builds the engine once and tests it from three languages.

```sh
cmake -B build_release -D USEARCH_DENSE_DTYPES=11
cmake --build build_release --target usearch_static usearch_shared
gradle -PusearchPrebuilt=$PWD/build_release test
```

Kotlin and Scala consume the same published Java surface rather than a binding of their own, and each has a suite beside the Java one in `java/test/cloud/unum/usearch/`, `InteropTest.kt` and `InteropTest.scala`, covering what only another JVM language can observe — that `@NullMarked` reaches a Kotlin caller as real non-null types, and that the array-based batch verbs take Scala arrays unconverted.
`check` runs all three.

```sh
gradle kotlinInteropTest scalaInteropTest
```

Adding a fourth JVM language is one entry in the `interopFrameworks` map in `build.gradle`, which derives its source set, its configurations and its test task.

## C#

Setup the .NET environment:

```sh
dotnet nuget add source https://api.nuget.org/v3/index.json -n nuget.org
```

USearch provides CSharp bindings on .NET 10, reaching the C interface through source-generated `LibraryImport`.
The project drives no CMake of its own, so the shared library is built first and dropped where the runtime probes for it.

```sh
cmake -B build_artifacts -D USEARCH_BUILD_SHARED=1
cmake --build build_artifacts --config Release -j
```

Then, on Windows, copy the library to the CSharp project and run the tests.
The binding imports `libusearch_shared`, which .NET probes for verbatim while MSVC emits no `lib` prefix, so the destination keeps the name the import expects:

```sh
mkdir -p ".\csharp\lib\runtimes\win-x64\native"
cp ".\build_artifacts\usearch_shared.dll" ".\csharp\lib\runtimes\win-x64\native\libusearch_shared.dll"
cd csharp
dotnet test -c Debug --logger "console;verbosity=detailed"
dotnet test -c Release
```

On Linux, the process is similar:

```sh
mkdir -p "csharp/lib/runtimes/linux-x64/native" # for x86
cp "build_artifacts/libusearch_shared.so" "csharp/lib/runtimes/linux-x64/native" # for x86
mkdir -p "csharp/lib/runtimes/linux-arm64/native" # for ARM
cp "build_artifacts/libusearch_shared.so" "csharp/lib/runtimes/linux-arm64/native" # for ARM
cd csharp
dotnet test -c Debug --logger "console;verbosity=detailed"
dotnet test -c Release
```

On macOS with Arm-based chips:

```sh
mkdir -p "csharp/lib/runtimes/osx-arm64/native"
cp "build_artifacts/libusearch_shared.dylib" "csharp/lib/runtimes/osx-arm64/native"
cd csharp
dotnet test -c Debug --logger "console;verbosity=detailed"
dotnet test -c Release
```

## Wolfram

USearch reaches the Wolfram Language through LibraryLink, whose boundary is `wolfram/lib.c`.
It builds from the repository root, and `USEARCH_WOLFRAM_PATH` names the Mathematica installation when it is not in the default place.

```sh
brew install --cask wolfram-engine

cmake -B build_release -D USEARCH_DENSE_DTYPES=11 -D USEARCH_BUILD_WOLFRAM=1 \
    -D USEARCH_WOLFRAM_PATH=/usr/local/Wolfram/Mathematica
cmake --build build_release --config Release --target usearch_wolfram

wolframscript -file wolfram/Test.wls
```

It links the C interface as an archive, so what Wolfram loads is one file, and it is the only target here without a `lib` prefix — Wolfram finds a library in `LibraryResources/$SystemID` by its bare name.

## Docker

```sh
docker build -t unum/usearch . && docker run unum/usearch
```

For multi-architecture builds and publications:

```sh
version=$(cat VERSION)
docker buildx create --use &&
    docker login &&
    docker buildx build \
        --platform "linux/amd64,linux/arm64" \
        --build-arg version=$version \
        --file Dockerfile \
        --tag unum/usearch:$version \
        --tag unum/usearch:latest \
        --push .
```

## WebAssembly

The WASI build uses the same module shape as the Emscripten `wasm32_v128_k` build, but for standalone runtimes and for the Rust crate's `wasm32-wasip1` target.
Unpack a [WASI SDK release](https://github.com/WebAssembly/wasi-sdk/releases) and point `WASI_SDK_PATH` at it; `cmake/toolchain-wasm32-wasi.cmake` names the compilers, the sysroot and the SIMD tier from there:

```sh
export WASI_SDK_PATH=/opt/wasi-sdk-34.0-x86_64-linux
cmake -B build_wasm32_wasi -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-wasm32-wasi.cmake -DUSEARCH_BUILD_SHARED=1 -DUSEARCH_BUILD_TEST=1
cmake --build build_wasm32_wasi --target usearch_static usearch_test_c
ctest --test-dir build_wasm32_wasi -R '^usearch_test_c$'
```

`ctest` runs the module under Wasmtime when it is on the path.
The Rust crate hands CMake the same toolchain file on its own, so `cargo build --target wasm32-wasip1` needs only `WASI_SDK_PATH`.

## Documentation

Sphinx builds the site under `docs/`.
A function's `@par Statuses:` block lists only its contract statuses, the conditions the caller controls or can observe, since the status enum documents once which categories any allocating, device-binding, file-touching or cancellable verb may also return.
Doxygen feeds the C and C++ references through Breathe, and the Python reference imports the compiled `usearch` package:

```sh
pip install -r docs/requirements.txt
pip install -e .                        # the package the Python reference documents
cd docs && doxygen conf.dox             # writes the XML into `build/xml`
sphinx-build -M html . ../build/docs -W # writes the site into `build/docs/html`
```

## Working on Sub-Modules

Extending metrics in NumKong:

```sh
git push --set-upstream https://github.com/ashvardanian/numkong.git HEAD:main
```
