# The C Interface Implementation

`c/usearch/*.h` declares the C ABI every binding consumes, and this directory implements it over the header-only engine in `include/usearch/`.
None of it is a public header: the C libraries, the C test and every binding link the objects these files compile into.

## Files

- `types.cpp` answers the library-wide probes of `types.h`: the version, the capability levels compiled and available, an outcome's status, and the spelling of every closed set in both directions.
- `lib.hpp` holds what every modality's dispatch shares: the CPU classes, the executor and arena handles, the deferred table, and the outcome, progress and output helpers.
- `dense_index.cuh` implements `dense_index.h`: the handle owning one picked variant through its residency's verb table, the ladders that pick it, and every exported verb as one visit over five residencies apiece.
- `dense_index.cpp` compiles that header with the host compiler for every build, and includes no tower.
- `dense_kernels.cuh` is what every dense GPU device unit compiles beside the kernels: the C value lookup's device call, the lookup a sharded attend walks through, and the ceiling of the class a pass compiles for.
- `dense_punned.cuh` is what the dispatch unit and the class units agree on, and includes no tower: the verb table `dense_verbs<residency>` one variant compiles, every entry taking `(index, inputs, outputs, config, executor, ticks)` as C++ composites, the `dense_class_tables<residency, class>` a class unit instantiates, and `punned_dense_index<residency>`, the picked table beside its `void*` index.
- `dense_specializations.cuh` is the one place dense variants are enumerated: the residency aliases, the static bodies of `specialized_index<indexes>` each table entry points at, and the fold over `dense_dtypes_t` that keeps the variants a class advances.
- `dense/ram_variants_<class>.cpp`, `dense/nvme_variants_<class>.cpp` and `dense/vram_variants_<class>.{hip,cpp}` are the class units, each one line instantiating `dense_class_tables` for its residency and class.
- `basic_punned.cuh` is what the basic dispatch units and class units agree on, and includes no tower: the crossings of keys and tapes, the verb table `basic_verbs<traits, residency>`, the `basic_class_tables<traits, residency, class>`, `punned_basic_index<traits, residency>`, and each modality's traits and geometry, with the composites each C argument structure lowers onto once in the dispatch unit; variants only return statuses, and only `*_index.cuh` writes outcomes.
- `basic_specializations.cuh` holds what the basic modalities' variants share: the fan-outs that insert and search through an executor, each residency's lookups and cursors, the GPU verbs every NVIDIA variant calls, and the static bodies of `specialized_variant<traits, variant>` each table entry points at.
- `sequence_specializations.cuh`, `shapes_specializations.cuh` and `geospatial_specializations.cuh` spell each modality's variants, such as `sequence_ram_variant<cpu_class>` and `shapes_ram_variant<scalar, cpu_class>`, with their NVMe and NVIDIA siblings, the `<modality>_variants_of` each class compiles per residency, and the `variant_bodies<traits>` that freeze, describe and place one variant.
- `basic_index.cuh` holds the one handle over a modality's picked variant, the CPU and GPU ladders, and every verb once, each modality's traits supplying only what differs.
- `sequence_index.cpp`, `shapes_index.cpp` and `geospatial_index.cpp` implement `sequence_index.h`, `shapes_index.h` and `geospatial_index.h`: the handle and one forward per verb.
- `arrow.cpp` implements `arrow.h`: Arrow columns read into views, and views exported as Arrow columns in place.
- `<modality>/cpu_variants_<class>.cpp` are their class units, each instantiating its modality's RAM and NVMe tables for the class, and `<modality>/vram_variants_nvidia.cpp` holds the modality's NVIDIA table beside a `vram_kernels_nvidia_<key>.cu` per metric or dtype, where geospatial's one `vram_variants_nvidia.cu` compiles both ways.

## From a Verb to an Engine Template

Every exported verb takes one argument structure and runs the same three steps:

```cpp
USEARCH_EXPORT void usearch_dense_search(usearch_dense_search_t* arguments) {
    if (submitted<usearch_dense_search>(arguments)) return;
    call_outcome_t outcome(arguments->outcome, arguments->arena);
    // ... validate the shapes, claim the arena, carve the outputs
    visit_punned(*arguments->index, [&]<vendored_residency_t residency_>(punned_dense_index<residency_> punned) noexcept {
        /* punned.verbs.search(punned.index, ...) */
    });
}
```

- `submitted<verb>` decides whether the call runs now or waits in the executor's deferred table, as described under [Deferred Calls](#deferred-calls).
- `call_outcome_t` marks the outcome pending, collects every step's status privately, and on unwinding releases the arena before the one final store.
  A path that forgot to report finishes `usearch_status_internal_k` rather than hanging its waiters.
- `visit_punned` reaches the table of the handle's residency, whose entries cast the handle's `void*` index back to its variant and forward to the engine's sharded index templates from `include/usearch/dense/`.

The variants are aliases over those templates, keyed by the storage dtype and the class:

- `ram_indexes<dtype, cpu_class>` → `dense::ram_indexes<uuid_t, quads_kit_of<dtype, cpu_class>, tiles_kit_of<...>, elementwise_kit_of<cpu_class>, rmscast_kit_of<...>, visits_kit_of<cpu_class>, waiter_of<cpu_class>, ...>`, each kit its ladder's pick under the class's NumKong mask, with the class's atomic reference and cache hints last.
- `nvme_indexes<dtype, cpu_class>` → `dense::nvme_indexes<uuid_t, ...>` over the same five kits.
- `vram_nvidia_indexes<dtype, gpu_class>` → `dense::vram_nvidia_indexes<uuid_t, dtype, cuda_fatbin<gpu_class, dtype>, fu::preferred_cache_hints_t>`, the fatbin of one dtype's kernels stating the class's NumKong mask as its `capability_k`, and `vram_amd_indexes<dtype, gpu_class>` likewise over `dense::vram_amd_indexes` and `hip_code_object<gpu_class, dtype>`.
- `vram_apple_indexes<dtype, gpu_class>` → `dense::vram_apple_indexes<uuid_t, dtype, numkong_capabilities_of(gpu_class)>`, one variant per Apple class, each compiling its own library at first use.

A variant's executor type is its residency, so `vendored_residency_of`, `is_ram_index`, `is_nvme_index` and `is_vram_index` read it rather than a second tag that could disagree.
Every residency fills every entry: an NVMe view answers `read_only_k` for each mutation, and a device variant `unsupported_k` for what only a host walk takes.
`insert` is one plan with a step per slice that publishes every entry at once, and refinements and cursors live behind a `void*` their own entries open, step and free.
Rows reach every variant in the caller's scalar as a `dense::rows_view_t`, and each variant imports them into its stored scalar per thread, in scratch its executor already owns, so the C layer never stages a row.
Scores reach the arena as the ABI's `double` straight from the engines, and row outputs export into the scalar the caller asks for.
A `usearch_progress_t` with a handler slices a batch into its `steps`, ten unless the caller asks otherwise, each handed to the engine whole and reported once; without a handler the call is one slice, and a GPU engine windows a large one through its own staging.
`progress_ticks_t` keeps that rule in one place, also thinning the ticks of verbs that count their own units, a checkpoint's shards or a refinement's nodes.

## Dispatch over Capability Kits

`init_ram`, `init_vram`, `load_*` and `view_nvme` lower the runtime geometry onto one variant:

1. `with_variant` maps the quantization onto a NumKong dtype, and the metric onto a kernel kind: the numeric dtypes speak inner, angular and euclidean, while the bit-packed `u1x8` speaks inner, hamming and jaccard.
   A dtype the build's `USEARCH_DENSE_DTYPES` left out of `dense_dtypes_t` answers `usearch_status_unsupported_k`.
2. `make_index` picks the class for the residency:
   - RAM walks `cpu_ladder_t` newest first, asking each compiled rung whose NumKong, StringZilla and ForkUnion capabilities the machine reports for its variant of the dtype, and ending on the serial floor.
     A monoversioned build's ladder holds its one `monoversioned` class, whose masks are the ones the libraries compiled.
     On x86 it reads `intel_diamond` → `intel_granite` → `intel_sapphire` → `amd_genoa` → `intel_icelake` → `intel_skylake` → `intel_sierra` → `intel_haswell`, on Arm `nvidia_vera` → `apple_m4` → `aws_g5` → `apple_m2` → `aws_g2`, then `alibaba_c930` on RISC-V and `wasm64_extended` → `wasm32_v128` on WebAssembly.
   - A class pinned in `usearch_ram_options_t::cpu_class` admits every rung whose NumKong, StringZilla and ForkUnion masks are subsets of its own, so pinning a class behaves as deriving on a machine of it.
   - NVMe walks the same ladder over the classes that compiled an NVMe unit; a storage walk shares a few counters and never waits, so a class advancing only by its waiter or reference compiles no NVMe unit and its parent serves.
   - VRAM reads the named USearch ordinals off the process device listing and takes the class every device shares, `nvidia-ampere` through `apple-metal10`; mixed vendors refuse with `config_mismatch_k`, mixed generations with `device_k`.

A CPU class is its parent plus a delta.
`numkong_capabilities_of` lowers it onto NumKong's cumulative compute mask, `parent_class_of` names the rung it degrades to, and `waiter_for_cpu_class`, `atomic_ref_for_cpu_class` and `cache_hints_for_cpu_class` pick its ForkUnion primitives.
A variant exists only where it advances past its parent: `variant_advances` compares the picked 4-way and tiles kits and the ForkUnion masks at compile time, and a class table without a variant of the dtype sends the walk to the next rung.
The handle records the class it picked once, at construction, which `usearch_dense_describe` reports as `cpu_class` on RAM and NVMe and `gpu_class` on VRAM.

## Compilation Units

The dispatch unit visits five residencies and compiles at the baseline instruction set, never seeing a tower.
Each class unit compiles its variants with the class's own flags into a table of function pointers, every body `static` or a static member of a template over the variant's own types, so no body compiled under a class's flags merges into a baseline caller.

- `usearch_dense_ram_variants` collects the CPU class units.
  `src/CMakeLists.txt` restates the lineage row for row: `usearch_dense_ram_variant(<class> <CLASS> OVER <parent> <kits>...)` unions the parent's flags with those of every delta kit that compiled, and the unit exists only when its parent does and some delta compiled, which the dispatch units read as `USEARCH_TARGET_<CLASS>`, the one gate of that class's variants in every list and ladder.
- `usearch_dense_nvme_variants` reuses those flags for the classes whose kits a storage walk tells apart, so `alibaba_c930` and the WebAssembly classes take none.
- A monoversioned build compiles `<modality>/<residency>_variants_monoversioned.cpp` alone in each CPU row, with the flags of every capability the libraries enabled.
- `usearch_dense_vram_variants_nvidia_<class>` embeds one NVIDIA class's fatbin per dtype, each compiled by NVCC device-only from `dense/vram_kernels_nvidia_<dtype>.cu` for the class's own architecture alone: `sm_80` with its PTX, `sm_89`, `sm_90a`, `sm_100f` and `sm_120f`.
  The class unit `dense/vram_variants_nvidia_<class>.cpp` holds the host variants, finding each dtype's fatbin in the table `dense_nvidia_<class>_bundles` CMake writes in `dense_dtypes_t` order, and the device units include `dense_kernels.cuh` rather than the specializations, whose CPU metrics would bring NumKong's own kernels into every pass.
  Nothing links `libcuda` or the CUDA runtime: `cuda_api_t::load` opens the driver through entry points it declares itself, once per process at the first VRAM call, and Clang's host-only pass builds the variants.
- `usearch_dense_vram_variants_amd_<class>` embeds one AMD class's code object per dtype, compiled for its own processors, and the CDNA3 floor for CDNA4 too.
  Its kernels sit in `dense/vram_kernels_amd_<dtype>.hip`, compiled device-only once per class beside the class unit, which names none of them.
  Nothing links `libamdhip64`: `hip_api_t::load` opens it through entry points it declares itself, so host units need no ROCm headers.
- `usearch_dense_vram_variants_apple` compiles the Apple variants as plain C++, one unit per Apple class, their kernels compiled from embedded source at first use.
  Nothing links Metal, Foundation or `libobjc`: `metal_api_t::load` opens them.
- `usearch_sequence_cpu_variants`, `usearch_shapes_cpu_variants` and `usearch_geospatial_cpu_variants` collect each modality's class units, `<modality>/cpu_variants_<class>.cpp`, each instantiating its RAM and NVMe tables, under the flags the dense rows computed for `intel_haswell`, `intel_skylake`, `aws_g2`, `apple_m2`, `alibaba_c930` and `wasm64_extended`.
- `usearch_<modality>_vram_variants_nvidia` embeds a basic modality's NVIDIA fatbins, one per sequence metric or shapes dtype and one for geospatial, every kit compiled past the Ampere floor.
- `usearch_punned` - `types.cpp`, `arrow.cpp` & the dispatch units, `dense_index.cpp`, `sequence_index.cpp`, `shapes_index.cpp` and `geospatial_index.cpp`.

`USEARCH_PUNNED_LIBRARIES` and `USEARCH_PUNNED_OBJECTS` hand all of them to `usearch_shared`, `usearch_static` and `usearch_test_c`, and the static archive carries the yyjson, NumKong and ForkUnion objects too, so a binding links one library.
`USEARCH_DENSE_DTYPES` trims every unit's variants from the front of `bf16`, `u1`, `i8`, `e4m3`, `f16`, `u8`, `e5m2`, `i4`, `u4`, `e3m2` and `e2m3`, keeping four by default.

## Executors & Arenas

Scratch belongs to the executor, never to the call.
Every buffer a verb only works through, imported rows, walk frontiers, tiles, softmax states, comes from where the verb runs: a thread's arena on a CPU, a slot's allocator on NVMe, a stream's staging on a GPU.
The engine running there sizes and reuses it, and the executor rewinds it once per verb.
A call's arena holds the outputs alone, the results that outlive the call, so neither the C layer nor an argument structure ever stages or allocates on a verb's behalf.

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

## Sequences & Shapes

A sequence or shapes handle holds one `basic::ram_indexes`, one shard per memory domain on the heap, with every value owned by its node record: a `sz::string` of the sequence's bytes, or an `nk::shapes_packed` of the shape's points in `f16` or `f32`.
Their ladders are shorter than dense's, over the classes where StringZilla's Levenshtein and overlap kits or NumKong's shapes and geospatial kits advance on their parent's.
Sequence, shapes and geospatial compile the same classes, each into a RAM and an NVMe unit, and a modality whose kits do not advance at a class falls back through its kit chain to the next lower generation.

- `insert` builds the batch's values on every thread from the index's allocator, sizes every thread's context and metric, and only then plans the batch, so nothing fails after the dispense.
- `search` writes each query's nearest from the walk's candidates, narrowed by the predicate a quad of keys at a time.
- Overlap distances are reported as the share of missed windows, from zero to one, rather than the kernels' fixed point.
- Checkpoints are one SafeGraphs monolith for a lone live shard, or one part per live shard beside a `.json` manifest, whose metric token is `levenshtein`, `levenshtein-utf8`, `needleman_wunsch`, `overlap`, `chamfer`, `hausdorff`, `gromov_wasserstein` or `wasserstein_procrustes`, with `connectivity`, `window_widths` and `points` in the provenance.

`refine`, `absorb`, `split` and `sample` serve RAM indexes, and a cursor opens at start keys and reports links there, while `view_ram` answers `usearch_status_unsupported_k`, as does every refinement on an NVIDIA index.
