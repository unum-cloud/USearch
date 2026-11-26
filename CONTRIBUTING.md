# Contributing to USearch

Thank you for coming here!
It's always nice to have third-party contributors 🤗

---

To keep the quality of the code high, please follow the conventions used across [Unum projects](https://github.com/unum-cloud):

- Each commit subject starts with `Add:`, `Break:`, `Chore:`, `Docs:`, `Fix:`, `Improve:` or `Make:`, as [Commits](#commits) details.
- Reference closed issues with `Closes #N` in the body, and credit collaborators with `Co-Authored-By:` trailers.

## Before You Start

Before building the first time, please pull `git` submodules.
That's how we bring in NumKong and other optional dependencies to test all of the available functionality.

```sh
git submodule update --init --recursive
```

The repository holds its own hooks, gating each staged file against the formatter its language uses and skipping the rest:

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

What the suites print, the variables they read, and how to write a case are under [Test Suites](#test-suites).

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
The rest match CI: `library` for the C library alone, `linux_arm64`, `android_arm64`, `android_armv7`, the three WASM toolchains, and `msvc2026_x64` / `msvc2026_arm64`.
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
    It holds the same library names in a `stub/` subdirectory, so pointing a binding at it is a search path rather than a different link line.
  - `USEARCH_CUDA_ARCH`, `USEARCH_ROCM_ARCH`, `USEARCH_METAL_ARCH` - build the NVIDIA, AMD or Apple variants and suites for the codes they name below, each empty and off unless a preset or `-D` names one
  - `USEARCH_DENSE_DTYPES` - how many dense storage dtypes the library, tests and benchmarks compile, from the front of bf16, u1, i8, e4m3, f16, u8, e5m2, i4, u4, e3m2, e2m3, `4` by default and `11` wherever a prebuilt package comes from: the `library`, `wasm` and `msvc2026_x64` presets, `pyproject.toml`, the npm scripts, `build.gradle`, `pixi.toml` and the per-binding recipes below. Binding suites skip the kinds a narrower build refuses.
    The C interface refuses a dtype the build left out with `usearch_status_unsupported_k`; the Rust crate builds from source and keeps the default four.
- Which CPUs to target:
  - `USEARCH_MARCH` - the floor of USearch's own units, seeded from the environment variable of the same name: empty for the architecture's floor, an ISA level like `x86-64-v3`, `native` for this host, or a CPU name like `graniterapids` or `neoverse-v2`, also when cross-compiling.
    Every class unit adds its class's flags over it, the tests and benchmarks drop their host tuning for it, and NumKong, StringZilla, ForkUnion and SmashTable take the same value.
  - `USEARCH_MULTIVERSIONED` - the build kind, `ON` by default and seeded from the environment too: a multiversioned build compiles every CPU class the toolchain compiles above the floor, picked at runtime, and with `OFF` a monoversioned build compiles one `monoversioned` class and no named class, `serial` included.
    The monoversioned class holds every capability NumKong, StringZilla and ForkUnion enabled: what the floor implies, plus what `-D <LIB>_TARGET_<CAP>=1` vouches for where no compiler macro reports it, like `FORKUNION_TARGET_ARM64_WFET=1` on Vera.
    Each modality and residency compiles one unit for it, and a handle whose `cpu_class_ceiling` sits below it, or run on a CPU lacking it, gets `usearch_status_unsupported_k`.
    `-D USEARCH_MARCH=native -D USEARCH_MULTIVERSIONED=OFF` is the native build, and the siblings take the same build kind.
- Which GPUs to target:
  - `USEARCH_CUDA_ARCH` - the NVIDIA code the library, tests and benchmarks compile for: `all` for the release set, `native` for each GPU `nvidia-smi` lists, mapped to its family code like `100f` on a B200 or B300, or one code like `90a`.
    A GPU class exists only when a code names it, `100`, `100a`, `100f`, `103a` or `103f` for datacenter Blackwell, so a monoversioned `100f` build compiles the Blackwell class alone, holding every modality, and refuses a code no class runs.
    A multiversioned build also always compiles the Ampere class, the floor, at `compute_80` PTX, which every GPU newer than any class runs, and which alone holds the sequence, shapes, geospatial and basic kernels, datacenter Blackwell adding its packed f16 shapes.
    The release set compiles each class at the lowest code its kernels need, as PTX alone where a plain code suffices, so every later GPU runs it, and never below CUDA 13's Turing floor: `compute_80` for the Ampere floor, whose kernels hold NumKong's Ampere `bf16x2` casts and `mma.sync`, `compute_89` for Ada's FP8, and `compute_120` for the RTX line.
    Hopper's `wgmma` and datacenter Blackwell's `tcgen05` need family codes, so their classes hold the `90a` and `100f` cubins instead.
    A native or named code compiles as SASS alone, with no PTX.
    NVCC compiles only device code, into fatbins the binaries embed; nothing links `libcuda` or the CUDA runtime, as the library opens the driver at first use.
    A NumKong GPU tier turns on in the CUDA tests only when every code runs it, so the release set exercises the Ampere one; iterate on one GPU with its family code, like `100f`, to test that generation's kits.
  - `USEARCH_ROCM_ARCH` - the AMD processor: `all` for gfx942, gfx950, gfx1250 and gfx1251, `native` for those `rocm_agent_enumerator` lists, or one like `gfx942`; each class embeds a code object per resolved processor it runs on, and a class with none does not exist.
  - `USEARCH_METAL_ARCH` - the Apple family: `all` for the `apple7`, `apple9` and `apple10` classes, or one of them, as Metal compiles the kernels on the device.
- Which other dependencies to use:
  - `USEARCH_WITH_JEMALLOC` - use Jemalloc for memory management
- How strict to be:
  - `USEARCH_WERROR` - treat compiler warnings as errors, as the `check` preset does

The sibling libraries hold no switches.
NumKong, ForkUnion, StringZilla and SmashTable are what the headers are written against, so every build acquires all four - from the submodule when a recursive clone provides one, and from the pinned commit otherwise.
The language bindings take their headers from the `numkong`, `forkunion`, and `stringzilla` trees in the checkout, so `pyproject.toml`, `package.json`, `build.rs`, and `Package.swift` each reach the same roots, and none of them builds from a non-recursive clone.

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

### SPDK Builds

The SPDK queue reads NVMe namespaces from user space, so it needs SPDK's static libraries, hugepages to DMA into, and a controller taken from the kernel's `nvme` driver.
Build SPDK once outside the checkout, and point `pkg-config` at its libraries and at the DPDK it bundles:

```sh
git clone --branch v26.09 --recursive https://github.com/spdk/spdk ~/Dependencies/spdk
cd ~/Dependencies/spdk && sudo scripts/pkgdep.sh && ./configure && make -j "$(nproc)" && cd -
export PKG_CONFIG_PATH=~/Dependencies/spdk/build/lib/pkgconfig:~/Dependencies/spdk/dpdk/build/lib/pkgconfig
cmake -D USEARCH_WITH_LINUX_SPDK=ON -D USEARCH_BUILD_TEST=1 -D CMAKE_BUILD_TYPE=RelWithDebInfo -B build_relwithdebinfo
cmake --build build_relwithdebinfo --target usearch_test usearch_tools
```

DPDK takes its memory from any mounted `hugetlbfs` whose page size matches pages the kernel reserved, so 1 GiB pages need nothing more than a reservation and a mount.
SPDK's setup script reserves 2 MiB pages unless told otherwise, so `SKIP_HUGE` leaves that to you, and `PCI_ALLOWED` limits it to the one controller under test, leaving every other drive with the kernel.
The SPDK suite runs in two passes around that binding: the first saves its part under `TMPDIR`, a copy places it on the namespace while the kernel still holds it, and the second views it through SPDK.

```sh
drive=nvme1n1
controller=$(basename "$(readlink -f /sys/block/$drive/device/device)")
USEARCH_FILTER=nvme_indexes build_relwithdebinfo/usearch_test
sudo build_relwithdebinfo/usearch copy "${TMPDIR:-/tmp}/usearch-nvme-spdk.safegraphs" /dev/$drive@64GiB

echo 4 | sudo tee /sys/kernel/mm/hugepages/hugepages-1048576kB/nr_hugepages
sudo mkdir -p /dev/hugepages1G && sudo mount -t hugetlbfs -o pagesize=1G none /dev/hugepages1G
sudo env SKIP_HUGE=yes PCI_ALLOWED=$controller ~/Dependencies/spdk/scripts/setup.sh config
sudo prlimit --memlock=unlimited:unlimited env USEARCH_FILTER=nvme_indexes \
    USEARCH_NVME_PATHS=spdk://$controller@64GiB build_relwithdebinfo/usearch_test
```

Release the controller and the pages as soon as the run ends, since a bound drive vanishes from `/dev` and reserved pages are memory no other process gets:

```sh
sudo env SKIP_HUGE=yes PCI_ALLOWED=$controller ~/Dependencies/spdk/scripts/setup.sh reset
sudo umount /dev/hugepages1G && sudo rmdir /dev/hugepages1G
echo 0 | sudo tee /sys/kernel/mm/hugepages/hugepages-1048576kB/nr_hugepages
```

### Capacities & Bounds

No cache, pool, selection, batch or buffer takes its size from a constant picked to fit, like `64` or `2 * connectivity`.
Each size follows from what its consumer holds, such as a walk's whole pool, or a row's live neighbors, one row of newcomers and its own row, or it comes from the caller.
Derive each bound once on the host and pass it to the kernels in the launch record, so that no kernel and no vendor tower recomputes it with a formula of its own.
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
When a choice has no derivation yet, derive it before committing it, from a cost model, a hardware property or a bound the algorithm guarantees, and state that derivation where the value is defined.
A value that can only be measured, like an empirical wave width, says so in its docstring, names what was measured, and lives in one definition every backend reads.

### Numeric Precision

Every score, distance and accumulator type comes from NumKong's per-input result types in `numkong/types.hpp`, read off the stored scalar or the kit wrapping it.
The dense tier stores `bf16`, `u1`, `i8`, `e4m3`, `f16`, `u8`, `e5m2`, `i4`, `u4`, `e3m2` and `e2m3`, whose inner products answer in `f32`, `i32` or `u32` through `dot_result_t`.
A sum over many of those products follows NumKong's reductions through `reduce_moments_sum_t`: `f32` for the floating formats, saturating `i64` and `u64` for the integers.
No dense computation needs `f64`, and widening to the largest type to be safe moves twice the bytes through every hot buffer.
The C interface's `double` `usearch_distance_t` is where scores land, never what they are computed in.

### Hot Entries

Every struct a thread keeps or moves along a hot path is flat and 8, 16 or 32 bytes, like `scored_node_t`, `nvme_flight` and `nvme_walk_payload`.
Lay its fields out directly rather than nesting structs, whose padding compounds, and pin its size with a `static_assert` beside the definition.
A would-be 24-byte entry shrinks first, through 32-bit node IDs or the kernel's own result type, and otherwise grows to 32 by a field that earns its place.

### Executors

An executor is its threads and their queues, one arena per thread, and its residency, and it serves one call at a time.
A `NULL` executor runs the call on the calling thread.
It creates no pool, thread, ring, SPDK queue pair or stream and probes no topology; its scratch and blocking queues live for that call only.
Name an executor for calls that allocate nothing once warm.
Such a call runs on `ram_executor_t::on_calling_thread()`, whose threads are an empty `fu::pool_t` and whose one arena serves that call alone; NVMe reads go through blocking queues over the files the index already holds, and VRAM work goes on each device's null legacy stream.
A VRAM executor owns one non-blocking stream per device, created with it, and caller-passed streams answer `unsupported_k`.
Tools and benchmarks open one executor and pass it to every call, and nothing outside `include/usearch/utilities/ram_device.hpp` dispatches on a raw `fu_pool_t`.
Every executor creates its own threads and pins them to the first cores the creating thread may run on, so a caller sharing a machine between executors narrows its affinity mask before creating each.
An NVMe executor serves RAM-resident handles through its threads too, so one executor drives a RAM index beside the views it reads.

No engine compares executor addresses to pick a path; the C layer may refuse an arena or plan used on another executor than its own in a one-line `if (... ->executor != ...)` guard, and `.githooks/pre-commit` refuses any other added `executor ==` or `executor !=` under `src/` and `include/`.
An index owns its media, never an executor: it keys its media list by a `media_list_key_t`, 64 bits from `media_identity_from_entropy()` drawn when it opens the list and the swap epoch its readers bracket, which a view leaves at zero.
An io_uring queue registers the list in its ring's file table only when the key differs from the one it holds, so a warm call registers nothing, and `nvme_begin_call` refuses a list with no identity as uninitialized.
The table keeps the descriptors of the last list registered, so the kernel keeps those files open after their index closes them, until the executor registers another list or is freed.
An NVMe executor opens its rings as deep as its paths' block queues admit and polled when every path polls; a deeper call reopens them once, and polled rings refuse media that do not poll.

Every RAM and NVMe executor caps each thread's arena at its `scratch_limit`, zero taking `scratch_limit_default_k`, and a call needing more answers out of memory.
A cursor keeps its state on the memory domain of the executor it was opened on.

### Threads and Placement

No mutable global or static lives in `include/` or `src/`, and `.githooks/pre-commit` refuses a non-`const` namespace-scope `inline` variable.
Who frees a resource, and which call uses it exclusively, are types such as `expected<>` owners and `claim_t`, never `bool` flags or raw pointers.
An executor creates its threads and their queues: one queue per thread, plus one at index `threads` for I/O run on the calling thread.
No verb creates a thread or a queue, and nothing runs that no one called.
Warm calls allocate nothing, since sizes come from hardware facts or the caller's budget, fixed when the executor is created.
A call with a `NULL` executor builds no pool, ring or SPDK queue pair; its blocking queues are made for it in `src/` and freed when it returns.
Spin only on completions the call itself submitted, and wait in the kernel for events a remote party triggers.
The library never names cores: threads and staging go on the memory domain the caller names, else the one the media sit on, inside the creating thread's affinity mask.
Every executor covers its own core set, so callers share a machine by giving each executor its own cores.
Residency is a type of the data handle: an executor of another residency answers `device_k`, a `NULL` executor on a residency whose queues `src/` cannot build per call, such as SPDK, answers `invalid_argument_k`, and one call runs per executor while a second answers `contended`.
One word names one concept in every U* library, and no word hands a thing over: say who owns it and until when it is valid.

### Hardening

`.githooks/pre-commit` holds the rules here that a check on added lines can hold.

- A `NULL` executor runs the call on the calling thread. It creates no pool, thread, ring, SPDK queue pair or stream and probes no topology; its scratch and blocking queues live for that call only. Name an executor for calls that allocate nothing once warm.
- No static, global, shared or thread-local pool, executor, queue set, stream or handle exists, stateless ones included: whoever needs one creates it and frees it.
- A `*_free` waits for the calls naming its handle and releases memory; it reads and writes no drive, and durability comes only from the flush verb.
- An executor owns its streams, created with it; a call without one uses the null legacy stream or is refused, and no second origin sits beside the owner.
- An engine creates, synchronizes and frees its own CUDA, HIP and Metal streams and allocates its own device and pinned memory; NumKong and StringZilla only provide kernels, which may still take an `nk_stream_t` or `sz_stream_t` handle.
- Every arena is freed before its executor; a binding enforces the order by lifetime where its language can and documents it otherwise, never by reference counts.
- A C struct change lands with every binding that lays the struct out by hand, checked field by field in review.
- Tests never restate a struct's layout that the compiler already pins, so no binding asserts a mirror's size or offsets.
- A handler the engine may call from several threads at once is safe to call concurrently in every binding: predicates and terrain callbacks.
- Every executor is equal; a handle owns its media, cache and epochs, never an executor, so no engine compares executor addresses to pick a path; the C layer may refuse an arena or plan used on another executor than its own, in a one-line `if (... ->executor != ...)` guard.
- Names say who owns a thing and until when a view is valid; no identifier in any language hands a thing over.
- A fix lands on every residency and vendor that has the bug, or its message names the ones left and why.
- A commit message names only what its diff shows.
- A wave never re-wraps or rewords comments it does not otherwise touch.
- A commit touching `.cu`, `.cuh`, `.hip` or `.metal` names the box that built and ran it, or says none did.

### Predicates

A predicate holds its cost, `usearch_predicate_cost_t` in `c/usearch/types.h`, and the engines ask it accordingly.
A cheap one is asked as candidates arrive, a full quad or a round's shorter tail at a static extent, with nothing buffered.
An expensive one gathers each thread's candidates beside the engine's payloads and answers them in one call per traversal round or scan run.
Those per-thread buffers are the only buffers a predicate ever holds, and an engine opens a per-thread family and reserves its buffers in the executor's arenas right after the call's rewind.

### Query Views

A batched kernel takes its queries as one flat transparent view over a static extent, like a `std::array` of views sized by the batch's extent, never as a span of pointers it chases one query at a time.
A query type that owns packed storage exposes a view of it, and the kernel reads that view.
`.githooks/pre-commit` refuses a `std::span` of pointers to const under `include/`, except a span of C strings, which names things, and a span of runs such as `std::span<run_t const* const>`, which lists the sorted runs a store merges; neither batches queries.

### Planned Operations

A long or multi-phase operation starts as a plan, like `plan_refine` or `plan_insert`, and its blocking verb is only a loop over a one-shot plan, on every residency.
Each operation has one plan type and one finish, like `usearch_<family>_<op>_plan_t` and `usearch_<family>_<op>_plan_finish`, in every family that offers the operation, and no verb describes a plan.
The plan is a lazy handle field on its verb: `NULL` runs the operation once, a `NULL` pointee opens a plan, and a live plan continues.
The finish frees the plan whatever it answers, and finishing a finished plan answers `invalid_argument`.
Cursors are not plans, and keep their `init`, `next` and `free` verbs; `next` refuses a `NULL` cursor or a foreign handle, and `free(NULL)` does nothing.
An engine plan owns its input and output lists, copying the `{count, handles}` pairs it was given, and the glue opens each operation through its own plan-open table entry, with no `std::variant` and no opener callable.
A free waits for the calls naming its single handle, which the library counts on the handle, so a binding keeps no counts of its own.
Every handle a list names, like a plan's sources and targets, must outlive the call and the plan's finish; the headers say so, and nothing counts the handles in a list.

Its parts are named by kind, and the kinds never mix.
A phase is a different kind of work, ended by an `advance()` barrier.
Phases are one `u8` enum shared by every plan, starting at `none_k = 0`, and mirrored by one C enum with `usearch_phase_name` and `usearch_phase_from_name`, with a `static_assert` per value where the glue casts it; progress reports `usearch_phase_progress_t{phase, ordinal, completed, total}` and no strings.
A repetition is the same work again, named for what repeats and counted, like a round or a depth.
A task is an independent unit of one phase that any thread runs in any order, ForkUnion's word; "step" names only sequential work.
`tasks()` lists the current phase's tasks, `run(task, context)` runs any of them on the calling thread, `advance()` is the barrier answering whether another phase follows, and `finish()` publishes and releases.
After the last phase, `advance()` keeps answering false and changes nothing.
Planning claims and reserves everything a task writes, so a task never fails for lack of memory and a plan stopped between tasks leaves the data correct.
Per-thread state lives in the traversal context `run` is handed.
An operation is a plan only when its work crosses a barrier between phases, runs long enough to stop and resume, or writes stored data; a pass whose threads each finish their own share is a direct verb over the caller's contexts.
A plan owns only what crosses its barriers: scratch belongs to the context, results land in the caller's outputs, named at planning, and whatever follows from other members is computed rather than stored.
`run`, `advance` and `finish` never allocate.
A report's verdict-like enum starts at `none = 0`, so a zeroed report never reads as progress.

USearch has two long operations, refine and reshard, each with its plan in dense and in every modality.
A refine round's phases are health, heal, hubs, gather and restitch.
A reshard names its sources and its targets: several sources into one target is a gather, one source into several targets is a cut, and one handle as its own only source and target is recut in place.
"Gather" names both the reshard of several sources into one target and the round's phase, which gathers entries into the neighborhoods under its hubs; both collect many into few, so they share the word.
It moves entries first, in phases of its own that rewrite links only within the parts it moves, then runs rounds, and no phase walks to find what an entry links to.
The phases are the values of `phase_t` in `utilities/types.hpp`, in order: copy, build, partition, scatter, health, heal, hubs, gather, restitch, write and done.
`search`, `multiply`, `attend` and MaxSim are direct verbs over the caller's contexts, and attention keeps its softmax states past the queries in `staged_queries`.

A member of a plan, cursor or walk names what it holds and when it holds it, like `picked_this_round_` or `visited_`.
A pair that trades places shares one stem and differs in one word, like `picked_this_round_` and `picked_next_round_`, and the verb moving between them is named for the barrier it runs at, like `advance_round_`.
No member is `bits`, `current`, `state`, `data`, `flags`, `tmp` or `scratch` alone, and none is named by its type.
Sizes come from the owner, never from a helper on the state.
Two bit runs are two `growing_bitset_t` members, never one buffer with an index.

### Refusals

A call the library cannot run answers one status, picked by what is wrong, in this order: `invalid_argument`, `read_only`, `config_mismatch`, `unsupported`, `contended`, `overflow`.
A `NULL` handle the call operates on answers `uninitialized` before any of them.
A malformed argument or list, one that is empty, `NULL` or names a handle twice, answers `invalid_argument`.
A write to a handle opened read-only answers `read_only`.
Handles of another kind, or of a geometry the call cannot join, answer `config_mismatch`.
A well-formed case nothing implements yet, like USearch's recut in place of a VRAM index under a shard capacity, answers `unsupported`.
An option the library cannot honour is refused with `unsupported`, never ignored.
A handle another call holds answers `contended` rather than waiting, and a size past its counter answers `overflow`.
A refusal moves nothing: every output and handle reads as it did before the call.
Each header lists a verb's refusals under `@par Statuses` in this order and in these words, and the stub library answers every one of them.

### Statuses

Statuses 0 to 19 keep the same numbers and meanings in every Unum library: success 0, deferred 1, out_of_memory 2, shape_mismatch 3, unsupported_layout 4, unknown_kind 5, read_only 6, io 7, device 8, malformed 9, internal 10, overflow 11, uninitialized 12, config_mismatch 13, contended 14, cancelled 15, unsupported 16, invalid_argument 17, not_found 18, already_exists 19.
A library leaves out the ones it cannot answer but never renumbers the rest, and numbers its own statuses from 20.
Enums pass as enum types; only the metrics and dtypes the CLI parses have a `_from_name`.
CPU and GPU class values are ABI: each class keeps the number USearch gives the same hardware, written explicitly, with unused numbers reserved rather than reused.

### Notes

A failure is `status_t {code, note}`, two words: recovery branches on the code, and the note, a literal or other static text, names the broken invariant, step or resource for logs and regression analysis.
Codes use the C interface's spellings and numbers, and the SDK in `cpp/` keeps an identical copy of `status_t`, so a status reads the same on both sides of the C layer.
A note is 2 to 8 words in sentence case with no closing period, never restates the code, and never carries a value that differs per call, such as a path, an error number, a size or an identifier.
A failure forwards whole, as `if (x.failed()) return x;`, a site that returns its own code writes its own note, and no site hides a callee's more specific failure behind a code of its own.
A call made once per candidate, neighbor, match or key may answer `expected<value_type_, status_code_t>`, the bare code, and the note is attached once where the loop's failure leaves it.
An atomic status word holds the code alone, and only the thread whose compare-exchange wins writes the note.
The C outcome's message is the note, else the rule of the call that was broken, else the status's own `usearch_status_name`.

### Compile-Time Choices

Code picks between alternatives at compile time through three gates, each for one kind of choice.

`#if` picks a toolchain or a platform, and nothing else.
A feature the compiler or the operating system lacks is a `#if`; a choice between two algorithms is not.

```cpp
#if defined(_WIN32) // allowed: the platform decides
using file_handle_t = void*;
#else
using file_handle_t = int;
#endif
```

A `requires` clause picks an overload when a whole function differs.
Every constrained overload has an unconstrained fallback beside it, so a type the constraint rejects still finds a function.

```cpp
template <typename index_type_>
status_t run_plan(index_type_& index) noexcept requires is_nvme_index<index_type_> { /* NVMe rounds */ }

template <typename index_type_>
status_t run_plan(index_type_& index) noexcept { /* every other residency */ }
```

`if constexpr` branches on a named concept or a `constexpr bool`, one level deep.
An `else if constexpr` chain is still one level.

```cpp
template <typename query_type_>
concept a_hinted_query = requires(query_type_& query) { query.find_hinted(0u); };

if constexpr (a_hinted_query<query_type_>) seeds = query.find_hinted(shard); // allowed: a named concept
else if constexpr (serial_class_k) seeds = {};                            // allowed: one chain, one level
```

The hook refuses each of these, in the directories its `gated_source_directories` names:

```cpp
if constexpr (requires { query.find_hinted(0u); }) { /* ... */ }   // refused: name the probe as a concept

if constexpr (is_ram_k) {
    if constexpr (is_serial_k) { /* ... */ }                        // refused: a second choice inside the first
}

template <typename value_type_, typename = std::enable_if_t<std::is_integral_v<value_type_>>>
value_type_ clamp(value_type_ value) noexcept;                      // refused: write `requires` on the overload

struct sorted_tag_t {};
void insert(entry_t entry, sorted_tag_t) noexcept;                  // refused: a tag that only picks an overload
```

Control flow nests at most three deep: `if`, `for`, `while`, `switch`, `do`, `try` and `catch` each count, and so does `if constexpr`.
An `else` or `catch` body sits at its `if`'s or `try`'s depth, so an `else if` chain stays flat.
A body without braces counts like a braced one, while a lambda or a plain block adds nothing.
Both sides of a `#if` count from the depth the `#if` began at.

```cpp
for (shard_t& shard : shards)                       // 1
    for (node_t node : shard.nodes)                 // 2
        if (node.live)                              // 3
            while (node.next != end) step(node);    // refused: 4 deep
```

To flatten, return early from failures, extract the inner loop into a function named for what it does, or split an overload per case.

The checks are `check_constexpr_conditions_named`, `check_constexpr_branches_flat`, `check_enable_if_and_tags_absent` and `check_constexpr_nesting`.
They judge only the lines a commit adds, so existing code flattens as it is touched.

### Split String Literals

A string literal split across lines by hand ends every fragment but the last with a trailing `//`.
Without the pin, clang-format joins the fragments or restacks them into a staircase.
This holds in C, C++, CUDA and HIP sources alike.

```cpp
char const help[] =                                 //
    "Usage: probe [options]\n"                      // allowed: every fragment but the last is pinned
    "Runs one probe.\n";

char const help[] = "Usage: probe [options]\n"
                    "Runs one probe.\n";            // refused: the first fragment is unpinned
```

The check is `check_split_string_literal_pinned`.
A comment that explains the line pins it just as well, and a bare `//` on a member line is not taken for a member's documentation.

### Components

Logic is never shared through inheritance: no base classes, CRTP or mixins.
Each residency and each GPU vendor is a dedicated tower with its own customizable logic.
Code two towers genuinely share becomes a component each tower owns as a member, or a free function each tower calls.
Host and device towers never fuse, since a RAM walk is a much finer-grained processing mode than a device batch.
A tower never takes a lock to make a shared design fit; it redesigns the protocol instead.

### Layers

`.githooks/pre-commit` checks the includes.

- `utilities/` includes itself, `usearch/types.h`, ForkUnion, NumKong and, in `*_device.hpp`, the vendor runtimes.
  It stops at hardware, memory, files and the SafeGraphs format: no item meaning and no CPU or GPU class.
- `basic/` includes itself and `utilities/`.
  It stops at a graph over opaque values, through the kits it is handed.
- `dense/` includes itself and `utilities/`.
  It stops at vectors and their kernels, through the kits it is handed.
- `sequence/`, `shapes/` and `geospatial/` include themselves, `basic/`, `utilities/` and their sibling library.
  Each stops at one value type's kits, and never includes dense or a `c/usearch` header.
- `cpp/usearch/` includes only `c/usearch/`, the C ABI in C++ clothes.
- `tools/` includes `c/usearch/`, `cpp/` and `utilities/`.
  Tools use the C ABI and never include an engine header.
- In every U* library, a file never includes both a library's SDK umbrella `<X/X.hpp>` and its native umbrella `<X.hpp>`, and nothing under `include/` includes a header that resolves under `cpp/`.
  C headers are the ABI every SDK sits on, so they are free everywhere.

### CPU Classes

Engines never reach for the standard library's waits or yields, and never hard-code a waiter.
Each template in `include/` takes its kits, cache hints, waiter and atomic reference as template arguments, and `src/` picks them per CPU class, as [Dispatch over Capability Kits](src/README.md#dispatch-over-capability-kits) describes.
An engine that needs richer behavior, like prefetching whole objects, wraps the primitive it was handed.
No template names a CPU class, an ISA instruction or a topology query, and capability masks stay in `src/`, where rungs are chosen.

### Dispatch Classes

The C enums give every class a fixed value, shared with UStore, UCall and UForm, so a ceiling value means the same hardware in all four.
CPU class 0 is `unknown`, which sets no ceiling so dispatch derives the CPU's own class, and GPU class 0 is `unknown`, which a device no class runs on reports.
The `cpu_class_ceiling` and `gpu_class_ceiling` options name the newest class dispatch may pick, and a description's `cpu_class` and `gpu_class` report the class it picked.

| Value | CPU class         | Adds over its parent                                                   |
| ----: | ----------------- | ---------------------------------------------------------------------- |
|     0 | `unknown`         | nothing: dispatch derives the CPU's own class                          |
|     1 | `serial`          | the portable floor                                                     |
|     2 | `monoversioned`   | everything the libraries compiled, in a monoversioned build alone      |
|     3 | `intel-haswell`   | AVX2 and FMA kits                                                      |
|     4 | `intel-sierra`    | AVX-VNNI and its successors over Haswell, TPAUSE waits, CMPCCXADD adds |
|     5 | `intel-skylake`   | AVX-512 kits over Haswell                                              |
|     6 | `intel-icelake`   | AVX-512 VNNI and VPOPCNT kits                                          |
|     7 | `amd-genoa`       | AVX-512 BF16 kits                                                      |
|     8 | `intel-sapphire`  | AVX-512 FP16 kits and AMX-BF16 and AMX-INT8 tiles                      |
|     9 | `intel-granite`   | AMX FP16 tiles                                                         |
|    10 | `intel-diamond`   | AVX10.2 FP8, the Sierra bundle, TPAUSE waits, CMPCCXADD adds           |
|    12 | `aws-g2`          | NEON kits                                                              |
|    13 | `apple-m2`        | dot-product, BF16 and FHM kits, LSE and RCpc atomics                   |
|    14 | `apple-m4`        | SME tiles                                                              |
|    15 | `aws-g5`          | WFE waits over Apple M2                                                |
|    16 | `nvidia-vera`     | FP8 dot products over Apple M2                                         |
|    17 | `alibaba-c930`    | RVV and Zvbb kits, RVV StringZilla kits, WRS waits, Zacas atomics      |
|    18 | `ibm-power10`     | VSX and MMA kits, `pause_short` waits                                  |
|    19 | `wasm32-v128`     | `simd128` kits                                                         |
|    20 | `wasm64-extended` | relaxed SIMD, shared memory and 64-bit addressing                      |

| Value | GPU class             | Unit codes in the release set                   |
| ----: | --------------------- | ----------------------------------------------- |
|     1 | `nvidia-ampere`       | `80-virtual`                                    |
|     2 | `nvidia-ada`          | `89-virtual`                                    |
|     3 | `nvidia-hopper`       | `90a-real`                                      |
|     4 | `nvidia-blackwell`    | `100f-real`                                     |
|     5 | `nvidia-blackwellrtx` | `120-virtual`                                   |
|     6 | `nvidia-rubin`        | none yet                                        |
|     7 | `amd-cdna3`           | `gfx942 gfx950`                                 |
|     8 | `amd-cdna4`           | `gfx950`                                        |
|     9 | `amd-cdna5`           | `gfx1250 gfx1251`                               |
|    10 | `apple-metal7`        | the Apple7 family, compiled on the device       |
|    11 | `apple-metal9`        | the Apple9 family, compiled on the device       |
|    12 | `apple-metal10`       | the Apple10 family, compiled on the device      |

Every compiled CPU class must advance past its parent for at least one dtype, and `dense_class_tables` asserts it at compile time.

### Pick Rules

`pick_cpu_ceiling` picks the CPU class once per handle and executor:

1. A value no class takes is refused with `invalid_argument`.
2. The machine's masks are the libraries' detected ones: NumKong's, StringZilla's and `fu::cpu_capabilities()`.
3. AMX is granted inside the pick, and a refused grant clears the AMX bits.
4. A `cpu_class_ceiling` intersects its masks with the machine's, so a class the machine lacks narrows the ceiling rather than being refused.
5. The ladder walk takes the newest compiled class whose masks fit the ceiling, and a ceiling no compiled class fits is refused with `unsupported`, as a monoversioned build on an older CPU is.

`pick_gpu_class` picks a GPU class per device, walking its vendor's ladder newest first:

1. The class must be built, as its gate `USEARCH_TARGET_<VENDOR>_<CLASS>` tells the dispatch units.
2. It must sit at or under the `gpu_class_ceiling`.
3. The driver must load the class's device code on the device, as `cuModuleLoadData` and `hipModuleLoadData` answer for its first dense bundle, or the device's Apple family must reach the class's.

Every device is listed, and one no class serves reports `unknown` and refuses work with `unsupported`.
An index whose devices pick different classes is refused with `config_mismatch`.

### Spelling Types

New code never names or sizes a type through `decltype`, `decltype(auto)` included.
Spell the type, or export it as a member alias where it is defined and read it from there, as `typename codec_t::view_t` names what a codec's `view` returns.
A `decltype` makes every reader evaluate an expression to learn a type its author already knew.
Templates take no `_t` or `_k` suffix, whether class, alias or variable templates or concepts, while concrete types end in `_t` and constants in `_k`.
A concept is a singular noun phrase naming what satisfies it, with `an_` before a vowel sound and `a_` otherwise, like `an_io_queue` or `a_vram_quads_kit`.

### Vocabulary

| Concept                                                          | Word                                  | Refused                                       |
| :--------------------------------------------------------------- | :------------------------------------ | :-------------------------------------------- |
| a kind of work in a plan, ended by a barrier                     | phase                                 | stage, step, pass                             |
| the same work again, counted                                     | repetition, named for what repeats: round, depth, nesting | phase, pass       |
| an independent unit of one phase, run by any thread in any order | task                                 | step, chunk                                   |
| sequential work                                                  | step                                  | task                                          |
| a per-capability kernel bundle                                   | kit                                   | trait                                         |
| a compiled alternative per class, dtype or residency             | variant                               | arm, cell                                     |
| RAM, VRAM or NVMe                                                | residency                             | home, memory side                             |
| an exported operation and its table entry                        | verb; trampoline                      | …                                             |
| a fixed-shape collection a verb fills                            | outputs                               | landing, lander                               |
| a member of a plan, cursor or walk                               | what it holds and when it holds it    | bits, current, state, data, flags, tmp, scratch |
| `*_index` in prose                                                                                  | mono-index                                                                                                      | singular, mono, standalone, single                                    |
| `*_indexes` in prose                                                                                | sharded index                                                                                                   | plural, group, collection, multi                                      |
| a mono-index inside a sharded index                                                                 | shard                                                                                                           | …                                                                     |
| one shard's graph in one file; several                                                              | monolith; parts under a manifest                                                                                | …                                                                     |
| which device or NUMA domain a shard sits on                                                         | placement                                                                                                       | …                                                                     |
| a residency split by the vendor its VRAM needs, one compiled variant each                           | vendored residency                                                                                              | home, tower                                                           |
| a stored item; its ID inside a shard; its ordinal in a batch                                        | entry; node; offset                                                                                             | row, value, member, item                                              |
| a horizontal slice of a physical matrix                                                             | row                                                                                                             | …                                                                     |
| one query and its results                                                                           | query                                                                                                           | row                                                                   |
| a dedicated per-residency or per-vendor implementation; its C-facing side                           | tower; face                                                                                                     | backend, engine class                                                 |
| the two long operations; the handles a reshard reads from and moves into                            | refine, reshard; sources, targets                                                                               | absorb, split, compact, compaction, merge, donor, receiver, recipient |
| a reshard of several sources into one target; of one source into several; of one handle into itself | gather; cut; recut                                                                                              | pack, absorb, merge, split, compact                                   |
| a callback consuming intermediate results                                                           | sink                                                                                                            | landing                                                               |
| the device fan-out's wait                                                                           | await                                                                                                           | join                                                                  |
| the nodes a refinement round re-selects                                                             | picked                                                                                                          | dirty, seam, marked                                                   |
| the nodes a walk has met and the ones it still has to expand                                        | visited, frontier                                                                                               | met, seen, queue                                                      |
| the reachability check: its verb, its plan and its phase inside a refine; what it returns           | health; reachability                                                                                            | check, scan, sweep, audit                                             |
| the phases of a refine or a reshard                                                                 | the same word in every plan: copy, build, partition, scatter, health, heal, hubs, gather, restitch, write, done | stage, audit, step, picking, sweeping, copying, reconnect, land       |
| what a sample returns; the points it starts from; the ones it keeps near                            | draws; shown; focus                                                                                             | pick, anchor, landmark, context, seed, pivot                          |
| a strong component outside the root's; one nothing links into; one with no link out                 | island; unreached island; stranded island                                                                       | orphan, remainder                                                     |
| an entry none of whose links is nearer to a target than itself                                      | dead end                                                                                                        | trap, local minimum, stall                                            |
| the entries a round draws to gather around; the one entry the health check starts from              | hubs; root                                                                                                      | centre, pivot, landmark, seed, region root, audit root                |
| the entries nearest one hub; the most one may hold, a core's cache                                  | neighborhood; neighborhood capacity                                                                             | region, bucket, cluster, cell, slice                                  |
| what a matrix engine multiplies at once                                                             | kit tile                                                                                                        | tile alone                                                            |
| what one round concludes                                                                            | improved, converged, stalled, stopped                                                                           | settled, done                                                         |
| device row storage                                                                                  | flat matrix, flat column                                                                                        | plane                                                                 |
| node ID width under `utilities/` and `basic/`                                                       | `node_id_type_`                                                                                                 | `node_id_t`, `u32_t node`                                             |
| the candidates a walk keeps                                                                         | candidates                                                                                                      | pool                                                                  |
| a bit run of a plan, cursor or walk | one `growing_bitset_t` member named for what it holds | bitset, buffer |
| how a batch's entries are shared among shards                                                       | split_batch_into_shards, offsets                                                                                | seat, deal, landed                                                    |
| who owns what a call reads | owner; the caller allocates and frees it | lent, lend, lends, lending, lender, borrow, borrowed, borrows, borrowing, loan, loans, loaned, lease, leases, leased |

`.githooks/pre-commit` refuses the last row's words in identifiers of every language, split at `_` and at camelCase, with comments and string literals blanked; Rust's standard `borrow`, `Borrowed` and `borrow_raw` names stay.

### Macro Naming

Every all-caps name starts with the full project name, `USEARCH_`.
A trailing `_` marks a name as internal: it may change in any release, and nothing outside this repository may define or test it.
A name without it is a public contract, either a switch you may set or a value you may read.

| Family                     | Form                             | Example                    |
| :------------------------- | :------------------------------- | :------------------------- |
| CPU or GPU class           | `USEARCH_TARGET_<CLASS>`         | `USEARCH_TARGET_SERIAL`    |
| Optional facility          | `USEARCH_WITH_[<OS>_]<FACILITY>` | `USEARCH_WITH_LINUX_URING` |
| Architecture or GPU vendor | `USEARCH_ARCH_<ARCH>_`           | `USEARCH_ARCH_CUDA_`       |
| Operating-system identity  | `USEARCH_OS_<OS>_`               | `USEARCH_OS_LINUX_`        |

Every name in these families is always defined, as 0 or 1, and tested with `#if`.
`defined(...)` appears only in a `!defined` default and in a guard that an `#error` follows.
A kernel facility, like `USEARCH_WITH_LINUX_URING`, defaults from the identity, and a dependency facility, like `USEARCH_WITH_LINUX_SPDK`, defaults to 0, and the target that links the dependency sets it.
Only the Platform Identity region of `c/usearch/types.h` reads operating-system predefines like `__linux__`.
Compiler predefines like `__clang__`, `__GNUC__` and `_MSC_VER` are read raw where they matter.

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

## Lean State

A type holds only what it cannot recompute, and names each part for what it holds.
These rules are shared by every sibling library, and the pre-commit hook enforces each rule that names a check.

- Members are named for what they hold, never for their type.
  Write `request->call` or `call->kept`, never `call->call` or `file.file`.
  Checked by `check_self_repeat_access`.
- Derived state is a method, not a field.
  A count, mask or capacity that follows from other members is computed when asked, so it can never disagree with them.
- One fact, one constant.
  Every `_k` constant states its derivation in its docblock, or says "measured" and where, and no other constant restates it.
- An owner that never moves holds its parts by value.
  A pointer is for a part that outlives its owner, moves apart from it, or is shared beyond it.
- Empty allocators take no bytes.
  An allocator member is declared `[[no_unique_address]]`, so a container over a stateless allocator is no larger than its own state.
  Checked by `check_empty_member_bytes`.
- No wrapper without its own state or invariant.
  A type that only forwards to one member is a name for that member, so use an alias or a function instead.
- Concepts are `a_`/`an_` noun phrases.
  A concept names what satisfies it, as `an_allocator` or `a_task_callback`: `an_` before a vowel sound, singular, with no `is_` or `has_` prefix and no `_concept` or `_t` suffix.
  Checked by `check_concept_names`.

## Test Suites

### Principles

Tests reach the library through its public surface: the C interface in `c/usearch/*.h`, the headers under `include/usearch/`, and each binding's own API.
The few private includes are listed under [Private Surface Waivers](#private-surface-waivers), each with what the public surface lacks.
A kernel is never its own oracle: results are held to NumKong's kernels in the index's own metric, to exhaustive scans, or to known answers, and inputs are never drawn through the code under test.
Oracles never compile out, so a test checks with `expect_true` and its siblings, never `assert`, and every C and C++ test unit opens with `#undef NDEBUG // ! A test's oracle must stay live in every build`.
Harnesses hold one mutable global, `test_run`, the run in progress.
Checks fire on worker threads and inside helpers as well as in a case's own body, so the run counts failed checks in one place and `run_test_case` diffs that count around each case, rather than threading a tally through every call.
After a crash, `p test_run` in a core dump names the case in progress and the last failed check, and the signal handlers print the case too.
The harness owns stdout, and case bodies write their diagnostics to stderr.
Settings are read once, before the first case, and a value that does not parse prints one line naming it and exits 1.
Skips are honest: a case that cannot run says why, as `needs <resource>`, `built without <SWITCH>`, `<facility> is refused on this machine` or `no <device>`, and a filter that selects nothing fails the run.
Benchmarks never include `test/`, each tree keeps its own harness, and neither uses a test or benchmark framework.

### Running

```sh
cmake --preset debug                    # Clang with AddressSanitizer
cmake --build --preset debug
ctest --preset debug
```

`usearch_test` holds the host suites, `usearch_test_vram` the GPU suites of every vendor the build holds, and `usearch_test_c` the C interface's.
Each opens by naming the build, the machine and the settings:

```text
USearch 3.0.0
- Compiled for: serial,neon,neonsdot,neonbfdot,neonfhm,neonfp8
- This machine: serial,neon,neonsdot,neonbfdot,neonfhm
- Seed: 42, scale: 1
- Filter: none
```

Both lists come from the C interface, the same strings the bindings report.
"Compiled for" is `usearch_capabilities_compiled()`, the NumKong capability levels this build holds kernels for, and "This machine" is `usearch_capabilities_available()`, those of them this processor can also run.
Every case then prints one verdict line, and a failed check prints where it failed, with what differed beneath it, while the case goes on, so one run lists every mismatch it reaches.
A failed case prints a `rerun:` line naming the seed, a filter that selects it alone, and the binary, and the run ends with a summary:

```text
  check failed at test/dense.cpp:214
    expected 10, got 9
- test_ram_metric<nk::f32_t> ... FAILED: 1 failed check(s)
  rerun: build_debug/usearch_test --seed 42 --filter '^test_ram_metric<nk::f32_t>$'
- test_insert_accepts_every_quantization<e2m3_t> ... skipped: built without e2m3 in USEARCH_DENSE_DTYPES
- test_search_finds_every_row ... ok (0.12 s)
41 passed, 1 failed, 1 skipped, seed 42
```

A run exits with 1 when a case failed, a check failed outside any case, or nothing ran, with 3 when `USEARCH_NO_SKIPS` is set and a case skipped, and with 0 otherwise.
The C suite prints the same lines, except that its `ok` holds no time and its `rerun:` line names only the seed, as it runs every case.
`USEARCH_TRAP_ON_FAILURE=1` raises `SIGTRAP` at the first failed check instead, which stops a debugger there, or dumps a core without one:

```sh
USEARCH_TRAP_ON_FAILURE=1 USEARCH_FILTER='^test_ram_' gdb -ex run --args build_debug/usearch_test
```

The bindings' suites run as their CI runs them:

```sh
python -X faulthandler -m pytest -x     # Python, after `uv pip install -e . --group test`
cargo test -- --nocapture               # Rust, showing the `skipped:` lines libtest otherwise hides
pixi run test                           # Mojo
```

pytest prints the same header, opening with `USearch <version> from <path>` and adding `- Dims:`, so a failure under a drawn seed reruns with the number it shows.

### Environment

| Variable                  | Flag                | Default                        | Meaning                                                                                               |
| :------------------------ | :------------------ | :----------------------------- | :---------------------------------------------------------------------------------------------------- |
| `USEARCH_SEED`            | `--seed`            | `42`                           | Seed every stream derives from, any unsigned 64-bit value, or `random` to draw one                    |
| `USEARCH_FILTER`          | `--filter`          | unset                          | C++ only: cases and benchmark rows to run, a regex searched in their labels                           |
| `USEARCH_SCALE`           | `--scale`           | `1`                            | Multiplier for every baseline iteration count in C++, like `0.1` or `10`; pytest prints it            |
| `USEARCH_TRAP_ON_FAILURE` | `--trap-on-failure` | unset                          | C++ only: raises `SIGTRAP` at the first failed check, to stop there under a debugger                  |
| `USEARCH_NO_SKIPS`        | `--no-skips`        | unset                          | C++ only: exits 3 when any case skipped, for a rig that must run everything                           |
| `USEARCH_DEVICES`         | `--devices`         | first device of every vendor   | C++ only: comma-separated USearch device ordinals, like `0` or `0,2`, as [Devices](#devices) explains |
| `USEARCH_NVME_PATHS`      | `--nvme-paths`      | unset                          | C++ only: comma-separated NVMe namespaces in the grammar of `usearch_nvme_options_t::paths`           |
| `USEARCH_DIMS`            |                     | `1,3,15,16,17,63,64,65,97,256` | Python only: the dimension counts the dense tests sweep, one count or a comma list                    |

A flag overrides its variable in the C++ suites and the benchmarks, as `--seed 7` or `--seed=7`; the C, Python, Rust and Mojo suites read the variables alone.
The C suite runs every case, pytest selects through its own `-k`, and the Rust suite reads only `USEARCH_SEED`, filtering through `cargo test <name>` instead.
The Mojo suite reads `USEARCH_SEED` as an integer, without `random`.
A value that does not parse stops the run with one line naming it, like `USEARCH_SEED="abc" does not parse, expected an unsigned integer or random`.

`USEARCH_SEED` keys every stream by a name, as `stream_key(seed, name, index)`, identical in C++, Python, Rust and Mojo, and the C++, Python and Rust suites name each case's streams after the case.
A case's inputs therefore never depend on which cases ran before it, so the seed together with the filter reproduces a failure:

```sh
USEARCH_SEED=random build_debug/usearch_test                         # A fresh seed, which the header prints
USEARCH_FILTER='^test_ram_' USEARCH_SCALE=0.1 build_debug/usearch_test   # A fast inner loop over the RAM cases
USEARCH_SEED=1528003704 python -m pytest test/ --count 3             # Replays that seed, then two more repeats
python -m pytest test/ -k quantization                               # Only the tests whose names mention it
```

### Writing a Test

A C++ case is a `void test_<behaviour>(args…)`, or a `case_result_t test_<behaviour>(args…)` when it may skip, which `main` registers as `run_test_case("<label>", test_<behaviour>, args…)`.
A label spells the code under test exactly, so grepping it lands on that code: a C ABI symbol like `usearch_init`, a qualified C++ name like `fu::flat_pool::for_n`, or the test function's own name like `test_open_clear_close`. A case run over a runtime choice is labelled in call notation with the exact constants, like `test_kit(row_kit_t::haswell_k)`, and one run over a template argument as written, like `test_metric_names<dense_verbs_t>`.
A case takes no settings or context parameter, since the run is global, and checks with functions from `test/harness.hpp`, each of which counts a failed check, lets the case go on, and returns whether it held:

- `expect_true(held)`, for any condition;
- `expect_equal(actual, wanted)`, which prints `expected <wanted>, got <actual>`;
- `expect_near(actual, wanted, tolerance)`, which holds when `|actual - wanted| <= tolerance * max(1, |actual|, |wanted|)`;
- `expect_status(outcome, wanted)`, which prints both statuses by name, and `expect_success(outcome)`, its shorthand for success;
- `expect_throws<E>(callable)`, whose subject is the failure.

A message trails the arguments as a format string and its values, like `expect_true(held, "row {} of {}", row, rows)`, and is never folded into the condition as `held && "text"`.
Where the step after a failed check would index past what it checked, the case stops on the check's answer, as in `if (!expect_equal(found.size(), wanted.size())) return;`.
A case that cannot run returns `case_result_t::skipped(reason)`.
Inputs come from `split_mix_generator_t {case_stream_key(index)}`, never from a literal seed, so a filtered run draws what an unfiltered one did.
Loops size through `scaled_iterations(baseline)`, waits go through `spin_until(predicate)`, bounded by `wait_limit_k`, and files go in a `scratch_directory_t`, removed with it.

A C case starts with `start_test_case("c.<case>")` and ends with `finish_test_case(NULL)`, or with its reason when it skipped.
It checks with `expect_(condition)`, or with `expect_at(held, __FILE__, __LINE__, "message")` where a message helps.

A Python test draws from the `seed` or `rng` fixture, never from a literal seed, passing `seed=seed` to `build_index` and `random_vectors`, and `stream_key(seed, "<family>", index)` where it needs several streams.
The seed is `stream_key(USEARCH_SEED, name, step)`, where `name` is the test function and its parameters, and `step` is the pytest-repeat step, so other parameters and repeats draw apart.
Axis names follow one vocabulary: `count` for the rows in a batch, `ndim` for a vector's length, `dtype` for the scalar type a caller hands in, and `quantization` for the one the index stores.
Parametrizes run top-down from `dtype` to `ndim` to `count` and then the rest, enum parameters are named by their lower-case members, and `MAX_CASES_PER_TEST` in `test/base.py` caps how many cases one test function expands to, so raising it is a decision for review.
A test that needs an optional dependency says so where it is decided, as `pytest.importorskip("pyarrow", reason="PyArrow is not installed")`, one needing a kernel family takes `@requires_capability("<codename>")`, and `init_ram_or_skip` skips a quantization the build's `USEARCH_DENSE_DTYPES` left out.
A test checking what a callback reported takes the `unraisable` fixture, which collects what `sys.unraisablehook` receives.

A Rust test is named as a behaviour sentence, without a `test_` prefix, returns `()` and calls `.unwrap()`, which reports the caller's line.
The helpers several modules' tests share live in `rust/testing.rs`: `skipped(reason)`, which a test that cannot run returns, `case_stream_key(index)`, which seeds its generators, `ScratchDirectory` for its files, `block_on` for its futures, and `assert_close` and `assert_close_rows`, which hold to the rule `expect_near` does.

A Mojo test is registered in `main` by hand, since Mojo has no test discovery, and one that is not silently never runs.
A test that cannot run calls `_skipped(reason)` and returns, `library_missing_k` being the reason when the shared library is not on the loader path, its generators start from `_stream_key(_run_seed(), "<family>", index)`, and its files go in a `mkdtemp()` directory.

One behaviour holds one name in every language: the Rust test's, which C, C++, Python and Mojo spell `test_<name>`, as in `saves_and_loads_a_file` and `test_saves_and_loads_a_file`.

### Benchmarks

The benchmarks under `bench/` keep their own harness, `bench/harness.hpp`, which reads its settings through the same pinned readers, so `USEARCH_SEED`, `USEARCH_FILTER`, `USEARCH_DEVICES` and `USEARCH_NVME_PATHS` mean there what they mean here, and the same flags override them.
Their rows take the label grammar of the tests, and they print the same header, while their timing tables keep their own layout.
`bench/dense.py` reads the same names, and [bench/README.md](bench/README.md) covers building, running and the datasets.

### Devices

`usearch_test_vram` runs on the GPUs `USEARCH_DEVICES` names, comma-separated USearch device ordinals like `0` or `0,2`, positions in the listing `usearch_gpu_devices` reports: NVIDIA devices first, then AMD, then Apple.
Unset, it takes the first device of every vendor present, and an ordinal past the listing prints the listing and exits 1.
It links one object library per vendor the build holds, and each device opens with its vendor's line: `- CUDA: <device> nvidia-<class> #<ordinal>, <N> SMs, <M> MiB`, `- ROCm: device #<ordinal> amd-cdna<N>` or `- Metal: device #<ordinal> apple-metal<N>`.
Without any GPU, every GPU case reports `skipped: no GPU device`.
A vendor switch, `USEARCH_ARCH_CUDA_`, `USEARCH_ARCH_ROCM_` or `USEARCH_ARCH_METAL_`, is a fact about the unit, so each takes its own `#if`, never an `#elif` after another vendor's, and anything in a shared header whose meaning depends on the vendor takes internal linkage or the vendor's name, since the shards link into one binary.

`USEARCH_NVME_PATHS` names NVMe namespaces the same way, comma-separated in the grammar of `usearch_nvme_options_t::paths`: files, block devices or SPDK transport IDs like `trtype:PCIe traddr:0000:01:00.0`.
Unset, the SPDK suite saves `usearch-nvme-spdk.safegraphs` under `TMPDIR` for a rig to copy onto a namespace, and the next run, naming that namespace, views the copy, as [SPDK Builds](#spdk-builds) shows.

### Private Surface Waivers

These test files include a header from `src/`, which no public header declares, each for what that surface lacks:

- `test/placement.cpp` includes `src/lib.hpp` for `resolve_placement`, `gpu_placement_t` and `gpu_class_bit`, the placement rules the C interface applies but does not export.
- `test/dense/vram_nvidia.hpp`, `test/shapes/vram_index.cpp` and `test/sequence/vram_index.cpp` include `src/lib.hpp` for `gpu_class_t`, `pick_gpu_class` and `stringzilla_capabilities_of`, the class the library picks a GPU unit by, which the cells compare against.
- `test/dense/ram_indexes.hpp` includes `src/dense_specializations.cuh` for `quads_kit_of` and its kin, the kits a CPU class runs a dtype on.

## Bindings

Every binding is a thin layer over the C interface, and these rules hold in each of them.
Typed bindings are those whose compiler checks callback signatures, and dynamic bindings are the rest; in USearch the typed ones are Rust, Zig, Mojo, Swift, C#, Java, Kotlin and Scala, Go and C++, and the dynamic ones are Python, JavaScript and Wolfram.

The C library decides what is valid.
A binding forwards its status, turns each status into its language's error in one place, re-checks nothing the library refuses, and keeps no copy of state the library already holds.
Name tables come from the library's `usearch_*_name` and `usearch_*_from_name` pairs, never from a switch written in the binding.

One mechanism reports every outcome: `usearch_outcome_status` reads it without waiting, and `usearch_outcome_wait` parks until it is final.
Nothing in a binding takes a lock, sends a notification or adds synchronization of its own.
The object that submitted a deferred call is the only one that polls its outcome, so no other thread ever waits on it, and whatever a second caller asks, the library answers.
A blocking `wait` calls `usearch_outcome_wait`.
An asynchronous bridge never parks a thread: it reads `usearch_outcome_status` and, while the call is deferred, yields to its runtime.
A runtime that can reschedule a task reads again on its next turn with no sleep, like Python's `loop.call_soon`, a C++ scheduler's `post(poll, state)` and Mojo's runtime, while one that resumes only on a timer pauses with a backoff growing from about 1 µs to 1 ms, like Swift's `Task.sleep`, JavaScript's `setTimeout` and C#'s `Task.Delay`.
A Rust future yields only: `poll` reads the status and, while it is deferred, calls `wake_by_ref` and returns `Poll::Pending`, with no sleep and no timer thread.

A callback never raises into the library.
Its fault never unwinds through C frames and never waits in a latch for the call to end: typed bindings give their callback types no way to throw, like a `noexcept` C++ callback type, while dynamic bindings report the fault through their language's unraisable path, like Python's `sys.unraisablehook` and JavaScript's uncaught-exception handler, and the callback answers cancel.
A Rust callback that panics aborts the process, as Rust does for any unwind reaching an `extern "C"` frame; nothing catches the unwind.
A binding that hands the library no callback of its own has nothing to report.

A view the library hands back lives until the next call on its arena.
The library describes where the memory sits, and the caller checks it; the binding neither copies the view to be safe nor checks its residency on the caller's behalf.

Every array a verb reads shares the call's one count, so mismatched shapes cannot be expressed.
Where a binding's own types hold two lengths, like a slice of keys beside a matrix of values, it checks once that they agree before the call crosses into the library, and nowhere else.

Each verb has one body, which fills the C struct and calls the library with no layers in between.
What it returns is read from the struct's output slot, a trampoline receives the handler itself as its state with no hub around it, and whether a call blocks or defers follows from the type of the receiver it is called on, never from a flag.
Verbs are never shared through a base class; a trait whose default methods hold the one body per verb is fine, like Rust's and Mojo's, since each receiver supplies only its handle, and C++ writes each handle and its receivers as one class template over what the receiver holds, so the handle carries the verbs as Rust's `&Handle` does.

A typed binding takes the scalar as a type parameter of each call, like `search::<f32>`, rather than letting a handle's type fix one.
A deferred Rust call owns what it passes the library until the call is final, since the borrow checker cannot see the library holding it.
A C++ wrapper over the C interface is a full typed binding: it has a deferred receiver, `noexcept` callback types and per-call scalar templates.

A dynamic binding publishes every field of every report the library fills.
A type the binding hands to C by cast is a transparent wrapper over the C struct, like Rust's `#[repr(transparent)]` newtype, so the cast is sound by construction and nothing asserts its size or alignment.
Tests never restate a struct's layout that the compiler already pins: no `offset_of!`, `size_of`, `MemoryLayout` or `Marshal.SizeOf` asserts against the C structs; `offsetof` tables that declare members are declarations, not tests.
Tests of how names print and parse, and of an engine run again through the binding, stay.
A binding test asserts a status, or the exception or error type mapped from it, and never the wording of a message or note, in any language: no pytest `match=`, no comparison of `error.message`, `err.Error()` or a note's text, and no substring search in them, since engines refine their notes over time.

A verb called on the handle itself runs on the calling thread, `on(executor)` blocks on an executor, and `deferred(executor)` submits there, with the same arguments in every language.
A deferred call allocates nothing in C++, Rust or Mojo: C++ and Mojo construct it immovable where the caller holds it and submit at once, while a Rust `Deferred` submits on its first poll or `wait`, once pinned in place.
A callback crosses as the C struct it lowers to, built by one generic constructor per handler shape with a trampoline made per handler type, never a trait object, a box or an enum.
A per-key predicate lowers as cheap, and a batch predicate as expensive unless the caller says it is cheap.
An arena never outlives the executor that made it: Rust's `Arena<'executor>` and Mojo's arena origin enforce the order, and the other bindings document it.

`.githooks/pre-commit` refuses locks in binding sources, throwing callback types, caught unwinds and kept exceptions.

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

The variables the suite reads, `USEARCH_SEED`, `USEARCH_SCALE` and `USEARCH_DIMS` among them, and the fixtures a test draws from are under [Test Suites](#test-suites).

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

To compile to WebAssembly make sure you have `emscripten` installed and pick one of the checked-in toolchains.
There is no WebAssembly target of its own — the C library is what crosses, and a module holds one SIMD tier, width and threads choice, so each toolchain file fixes all three:

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

The helpers the tests share and the seed they read are under [Test Suites](#test-suites).

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

Mojo has no test discovery, so every check is registered in `main` by hand and one that is not silently never runs, as [Test Suites](#test-suites) describes.

## Swift

USearch provides Swift bindings through the [Swift Package Manager](https://swiftpackageindex.com/unum-cloud/USearch).
Swift 6.4 or newer is required, as `Package.swift` declares.

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
It covers what the binding itself decides — key byte order, options marshalling through the residency union, matrix lowering, handle lifetime and the span views — since that vector search returns near neighbors is the engine's contract and is tested where the engine lives.

Formatting is enforced with Apple's `swift-format`, whose style lives in the `.swift-format` file at the repository root:

```sh
swift format --in-place --recursive swift Package.swift
swift format lint --strict --recursive swift Package.swift
```

Neither the build nor the suite needs an Apple device — both run on Linux, where [`swift.org/install`](https://www.swift.org/install) holds the toolchains.
A toolchain built for an older Ubuntu than the host may want that release's `libxml2` and `libicu` alongside it, since it links their older sonames.

## Go

The Go module sits at the repository root, so cgo reads the headers straight from `c/` and links the static archive the C build produces.
Compile the C library first, point cgo at it, and run everything from the root.

```sh
cmake -B build_release -D USEARCH_BUILD_SHARED=1 -D USEARCH_DENSE_DTYPES=11
cmake --build build_release --config Release --target usearch_static -j

export CGO_LDFLAGS="-L$PWD/build_release" # the archive `golang/types.go` links, NumKong and ForkUnion included
go test -race -v ./golang/...
```

The race detector matters here: the tests drive one shared handle from many goroutines, and it is the only thing that catches a missing `runtime.KeepAlive` around a matrix crossing into C.
Name `./golang/...` rather than `./...`, which would also walk `node_modules`.

Compilation and the static checks need only the headers, not the library, so they run before it links:

```sh
go build ./golang/...      # compiles the cgo preamble and shim.c against the real headers
go vet ./golang/...        # catches the cgo pointer-rule violations the headers cannot
gofmt -l golang            # must print nothing
staticcheck ./golang/...   # if installed
golangci-lint run ./golang/... # if installed
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

unum.cloud renders the reference from a symbol bundle that every release holds as `usearch_docs_<version>.tar.gz`.
A function's `@par Statuses:` block lists only its contract statuses, the conditions the caller controls or can observe, since the status enum documents once which categories any allocating, device-binding, file-touching or cancellable verb may also return.
The bundle holds each binding's own tool output: Doxygen XML for C, C++, Java and C#, griffe for Python, rustdoc JSON, a Swift symbol graph and `mojo doc`, plus the sources of the rest:

```sh
.github/scripts/build-docs.sh # writes build/usearch_docs_<version>.tar.gz
```

It needs Doxygen, `griffe`, the Rust toolchain the script names, Swift and Mojo on the path.
Swift typechecks against NumKong's module from the `numkong` submodule, or from `FETCHCONTENT_SOURCE_DIR_NUMKONG`.

## Commits

`.githooks/commit-msg` checks every message:

- The title reads `<Verb>: <Summary>`, where the verb is one of `Add`, `Break`, `Chore`, `Docs`, `Fix`, `Improve` or `Make`.
- The summary opens with a capital letter, unless it opens with a code identifier, which keeps its own case.
- The title runs to at most 80 characters, and a body line to at most 100 columns, unless it holds a URL or is indented code.
- The message holds no AI-assistant attribution, since authorship is the human who committed.

Git's merge, `fixup!` and `squash!` titles are exempt from the title rules.
A `Release: vX.Y.Z` title is exempt too, but its commit must stamp every version site, and no other commit may change `VERSION`.
`.githooks/pre-commit` formats and judges the staged lines, and `.githooks/selftest` runs both hooks against their fixtures.
On a pull request, `.github/scripts/check-commits.sh` replays `commit-msg` over every commit and runs `pre-commit` over the whole range.

## Releases

`.github/scripts/mint-release.sh X.Y.Z` cuts a release from a clean checkout of the branch it ships.
It builds the Swift binaries for that version, stamps every version site and the binaries' checksums, and commits `Release: vX.Y.Z`.
`.github/scripts/check-release.sh` then checks that commit, even in a clone where `core.hooksPath` is unset, before the script tags it and pushes both.
Then it publishes the GitHub release with the binaries attached, marking `X.Y.Z-rc.N` candidates as prereleases.
Publishing starts every `release-<binding>.yml` at once, and each runs `check-release.sh` before it builds anything, so each registry trusts only its own binding's file.

## Working on Sub-Modules

Extending metrics in NumKong:

```sh
git push --set-upstream https://github.com/ashvardanian/numkong.git HEAD:main
```
