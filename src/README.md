# The C Interface Implementation

`include/usearch/*.h` declares the C ABI every binding consumes, and this directory implements it over the header-only engine in `include/usearch/`.
None of it is a public header: the C libraries, the C test and every binding link the objects these files compile into.

## Files

- `types.cpp` answers the library-wide probes of `types.h`: the version, the capability levels compiled and available, an outcome's status, and the spelling of every closed set in both directions.
- `dense_index.cuh` implements `dense_index.h`: the variant grid over every compiled arm, the handle owning the chosen one, and every exported verb as one visit apiece.
- `dense_index.cpp` compiles that header with the host compiler for every build, the VRAM arms joining the variant for each vendor built.
- `dense_specializations.cuh` spells one arm: the scalar and capability tables, the CPU and GPU classes with the ForkUnion primitives they pair, and the `specialized_index` view every class unit instantiates.
- `dense/ram_variants_<class>.cpp`, `dense/nvme_variants_<class>.cpp` and `dense/vram_variants_<class>.{cu,hip,cpp}` are the class units, each instantiating `specialized_index` over its class's arms, one per storage dtype the class advances.
- `sequence.cpp` instantiates the sequence checkpoint verbs over the default index, so they keep compiling until the sequence modality's entry points land.
  So far `sequence_index.h`, `matrix_index.h` and `geospatial_index.h` are answered only by the brute-force stand-in in `src_stub/`.

## From a Verb to an Engine Template

Every exported verb takes one argument structure and runs the same three steps:

```cpp
USEARCH_EXPORT void usearch_dense_search(usearch_dense_search_t* arguments) {
    if (submitted<usearch_dense_search>(arguments)) return;
    call_outcome_t outcome(arguments->outcome, arguments->arena);
    // ... validate the shapes, claim the arena, carve the outputs
    std::visit([&]<typename variant_type_>(variant_type_& variant) noexcept { /* specialized_index {variant} */ },
               arguments->index->variants);
}
```

- `submitted<verb>` decides whether the call runs now or waits in the executor's deferred table, as described under [Deferred Calls](#deferred-calls).
- `call_outcome_t` marks the outcome pending, collects every step's status privately, and on unwinding releases the arena before the one final store.
  A path that forgot to report finishes `usearch_status_internal_k` rather than hanging its waiters.
- `std::visit` over `usearch_dense_index_handle_t::variants` reaches the arm, and `specialized_index {variant}` forwards to the engine's multi-shard templates from `include/usearch/dense/`.

The arms are aliases over those templates, keyed by the storage dtype and the class:

- `ram_indexes<dtype, cpu_class>` → `dense::ram_indexes<uuid_t, dtype, numkong_capabilities_of(cpu_class), waiter_of<cpu_class>, ...>`, with the class's atomic reference and cache hints last.
- `nvme_indexes<dtype, cpu_class>` → `dense::nvme_indexes<uuid_t, dtype, numkong_capabilities_of(cpu_class)>`.
- `vram_nvidia_indexes<dtype, gpu_class>` → `dense::vram_nvidia_indexes<uuid_t, dtype, numkong_capabilities_of(gpu_class)>`, and `vram_amd_indexes<dtype, gpu_class>` likewise over `dense::vram_amd_indexes`.
- `vram_apple_indexes<dtype>` → `dense::vram_apple_indexes<uuid_t, dtype>`, one arm serving every Apple class until a family's kernels differ.

An arm's executor type is its residency, so `is_ram_index`, `is_nvme_index` and `is_vram_index` read it rather than a second tag that could disagree.
The core verbs land on `specialized_index` members of the same names, `count`, `get`, `reserve`, `clear`, `refine`, `health`, `search`, `multiply`, `maxsim`, `attend`, `save` and `load`.
`insert` becomes `add`, one plan with a step per chunk that publishes every row at once.
Rows whose scalar differs from the stored one cast through NumKong into a staged matrix, 4096 rows at a time, while matching rows reach the arm in place.
Progress callbacks tick once per such chunk, and scores staged at the engine's width widen to the ABI's `double` as they land in the arena.

## Dispatch over Capability Kits

`init_ram`, `init_vram`, `load_*` and `view_nvme` lower the runtime geometry onto one arm:

1. `with_cell` maps the quantization onto a NumKong dtype.
   A dtype the build's `USEARCH_DENSE_DTYPES` left out answers `usearch_status_unsupported_k`.
2. `with_metric` maps the metric onto a kernel kind: the numeric dtypes speak inner, angular and euclidean, while the bit-packed `u1x8` speaks inner, hamming and jaccard.
3. `make_index` picks the class for the residency:
   - RAM descends a ladder of CPU classes newest first, taking the first rung whose NumKong and ForkUnion capabilities the machine reports and ending on the unconditional serial floor.
     On x86 it reads `intel_diamond` → `intel_granite` → `intel_sapphire` → `amd_genoa` → `intel_icelake` → `intel_skylake` → `intel_sierra` → `intel_haswell`, on Arm `nvidia_vera` → `apple_m4` → `aws_g5` → `apple_m2` → `aws_g2`, then `alibaba_c930` on RISC-V and `wasm64_extended` → `wasm32_v128` on WebAssembly.
   - NVMe walks the same classes, WebAssembly aside, and compares only kits, since a storage walk carries no waiter or atomic reference.
   - VRAM reads the named USearch ordinals off the process device listing and takes the class every device shares, `nvidia-ampere` through `apple-metal10`; mixed vendors refuse with `config_mismatch_k`, mixed generations with `device_k`.

A CPU class is its parent plus a delta.
`numkong_capabilities_of` lowers it onto NumKong's cumulative compute mask, `parent_class_of` names the rung it degrades to, and `waiter_for_cpu_class`, `atomic_ref_for_cpu_class` and `cache_hints_for_cpu_class` pick its ForkUnion primitives.
A rung exists only where it advances past its parent: `rung_advances` compares the picked 4-way and tiles kits and the ForkUnion masks at compile time, and a `static_assert` fails the build while the variant still lists an arm that stopped advancing.
The handle's capability string names the picked kit once, at construction, which is what `usearch_dense_describe` reports.

## Compilation Units

The dispatch unit visits every arm yet compiles at the baseline instruction set.
`extern template struct specialized_index<...>` declarations keep every arm's body out of it, and each class unit compiles its arms with the class's own flags.

- `usearch_dense_ram_variants` collects the CPU class units.
  `src/CMakeLists.txt` restates the lineage row for row: `usearch_dense_ram_variant(<class> OVER <parent> <kits>...)` unions the parent's flags with those of every delta kit that compiled, and the unit exists only when its parent does and some delta compiled.
- `usearch_dense_nvme_variants` reuses those flags for the classes whose kits a storage walk tells apart, so `aws_g5`, `alibaba_c930` and the WebAssembly classes take none.
- `usearch_dense_vram_variants_nvidia_<class>` embeds one NVIDIA class's fatbin, compiled by NVCC device-only for its own architecture alone: `sm_80` with its PTX, `sm_89`, `sm_90a`, `sm_100f` and `sm_120f`.
  Nothing links `libcuda` or the CUDA runtime: `cuda_api` opens the driver at first use through entry points it declares itself, and Clang's host-only pass builds the arms.
- `usearch_dense_vram_variants_amd_<class>` embeds one AMD class's code object, compiled for its own processors, and the CDNA3 floor for CDNA4 too.
  Nothing links `libamdhip64`: `hip_api` opens it at first use through entry points it declares itself, so host units need no ROCm headers.
- `usearch_dense_vram_variants_apple` compiles the Apple arms as plain C++, their kernels compiled from embedded source at first use.
  Nothing links Metal, Foundation or `libobjc`: `metal_api` opens them at first use.
- `usearch_punned` - `types.cpp`, `sequence.cpp` & the dense dispatch unit, `dense_index.cpp`.

`USEARCH_PUNNED_LIBRARIES` and `USEARCH_PUNNED_OBJECTS` hand all of them to `usearch_shared`, `usearch_static` and `usearch_test_c`, and the static archive carries the yyjson, NumKong and ForkUnion objects too, so a binding links one library.
`USEARCH_DENSE_DTYPES` trims every unit's arms from the front of `bf16`, `u1`, `i8`, `e4m3`, `f16`, `u8`, `e5m2`, `i4`, `u4`, `e3m2` and `e2m3`, keeping four by default.

## Executors & Arenas

`usearch_executor_handle_t` holds the topology a RAM or NVMe executor was spawned over, its deferred table, and one residency's executor.
The executor is a `std::variant` of `forkunion_executor_t`, `nvme_executor_t` and, per GPU vendor built, `cuda_executor_t`, `hip_executor_t` or `metal_executor_t`, whose alternative is the kind tag.
A verb whose executor kind does not match the handle's residency answers `usearch_status_device_k`.
A null executor builds one for that call alone: the calling thread on RAM, one thread of blocking reads on NVMe, and the default streams of the index's devices on VRAM.

`usearch_arena_handle_t` owns a growing arena of host pages and, under CUDA, one of device memory on the first device of the VRAM executor it was created with.
`claim_arena` holds it for one call and rewinds it, freeing the previous call's outputs, after refusing:

- a null arena, with `usearch_status_invalid_argument_k`;
- an arena on another device than the call computes on, with `usearch_status_device_k`;
- an arena another call holds, with `usearch_status_contended_k`;
- an input that lies inside the arena, with `usearch_status_invalid_argument_k`.

`carve` then hands out the call's outputs, and `call_outcome_t` lets go of the arena just before the final store, so a waiter that sees the status may reuse it at once.

## Deferred Calls

A call naming an executor and `usearch_completion_deferred_k` runs nothing at submission:

- `submitted<verb>` copies its argument structure by value into a free slot of the executor's table, tickets it, holds its arena, and returns with the outcome pending.
- The table is sized when the executor is created, 64 slots unless `deferred_limit` names another count, so a submission never allocates and a full table finishes the call `usearch_status_contended_k`.
- A held call runs on whichever thread first needs it, oldest ticket first: `usearch_outcome_wait` on its outcome, `usearch_executor_drain`, a blocking call on the same executor, or freeing any handle it names.
  Every call held before it on that executor runs first, so the executor keeps its submission order.
- `deferred_registry` lists every executor under one mutex and condition variable, which is how a wait or a free finds the calls naming its handle without asking each executor.
- The runner passes the copied arguments back into their verb through the thread-local `resuming_call`, which runs them in place rather than holding them again.

## NVMe

`usearch_dense_view_nvme` reads the checkpoint's header, adopts its geometry and metric, and opens every part in place, optionally on one device or directory per part.
The result is a read-only `dense::nvme_indexes` view, so every mutating verb answers `usearch_status_read_only_k`.
`usearch_executor_for_nvme` spawns threads that each open an I/O engine keeping 64 reads in flight.
Kernel paths take the first engine the host admits, from `io_uring` through Linux AIO to blocking reads.
Transport identifiers such as `trtype:PCIe` need an SPDK build, `USEARCH_WITH_LINUX_SPDK`, with hugepages for PCIe controllers alone, and one executor never mixes them with kernel paths.
The NVMe class units score with the same kits as the RAM ones, since the transport is the executor's runtime choice rather than a template parameter.
