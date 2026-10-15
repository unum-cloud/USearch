/**
 *  @file c/usearch/types.h
 *  @author Ash Vardanian
 *  @date August 14, 2026
 *  @brief Shared vocabulary of the USearch C interface: scalar types, keys, distances, shapes,
 *      outcomes, executors, arenas, and library introspection.
 *
 *  Everything the modality headers have in common - @c dense_index.h, @c sequence_index.h,
 *  @c shapes_index.h, and @c geospatial_index.h - and nothing they don't. Include @c usearch.h for
 *  all of them, or a modality header alone, since each pulls this one in. The C vocabulary sits
 *  beside its C++ twin @c types.hpp, so one installed tree serves both languages and the headers
 *  pair by stem.
 *
 *  @section types_verbs Verbs
 *
 *  Every verb that takes options, outputs, or callbacks takes one argument structure named after
 *  it - @c usearch_dense_search takes a @c usearch_dense_search_t - so optional arguments are
 *  skipped by name and their order never matters. Name lookups and the @c _free family stay
 *  positional. Every structure opens in the same order: the handle, @c executor, @c outcome, then
 *  @c completion. Inputs follow, then outputs and the @c arena they land in, then knobs, and last
 *  @c progress and @c stats.
 *
 *  Structures carry no size or version field, so they grow source-compatibly only: initialize
 *  them by designator, which zeroes every field left out, and expect the major version to change
 *  whenever one does.
 *
 *  @section types_asynchrony Asynchrony
 *
 *  Every call names its @c completion, and synchronous and deferred calls share one shape.
 *  A blocking call, the default, returns with its final status stored and every output readable.
 *  A deferred call returns once submitted, as an io_uring entry does, and finishes on its executor.
 *  A call with a @c NULL executor runs on the calling thread and is final when it returns.
 *  Submission copies the structure; everything it points to stays borrowed until final.
 *
 *  - @c outcome->status reads @c usearch_status_deferred_k until final, even for a call refused at
 *    submission.
 *  - The message, then the final status, are stored with release ordering and a wake.
 *  - The caller reads the status with acquire ordering, polling through @c usearch_outcome_status
 *    or waiting through @c usearch_outcome_wait.
 *  - A GPU executor stores the final status from a host callback that still fires after a device fault.
 *  - Executors own their streams, so a caller orders dependent calls by waiting on the first.
 *  - A submission finding the deferred table full, sized by @c deferred_limit, finishes @c usearch_status_contended_k.
 *  - A call stops early only when its progress handler returns @c usearch_cancel_k.
 *
 *  @code{.c}
 *  usearch_outcome_t outcomes[4] = {0};
 *  for (size_t shard = 0; shard != 4; ++shard) {
 *      searches[shard].outcome = &outcomes[shard];
 *      searches[shard].completion = usearch_completion_deferred_k;
 *      usearch_dense_search(&searches[shard]);
 *  }
 *  for (size_t shard = 0; shard != 4; ++shard) usearch_outcome_wait(&outcomes[shard]);
 *  @endcode
 *
 *  @section types_memory Memory
 *
 *  Two complications shape memory, and nothing else may: outputs whose size only the backend
 *  knows, and memory a GPU should place itself. Arenas answer the first, executors the second, and
 *  every binding stays one verb per operation rather than a twin per output style. Every byte has
 *  one owner, and each owner picks where its bytes live once, when it is created:
 *
 *  - Callers lend memory, never a way to allocate it. No allocator callback crosses the interface,
 *    and every pointer a caller passes is an input, classified through the driver once per call.
 *  - Payload belongs to the handle, placed by its constructor's residency, as @c usearch_dense_init_ram
 *    places it in host memory.
 *  - Scratch and staging belong to the executor, placed at its creation and grown to a high-water mark.
 *  - Every output whose length scales with the call lands in an arena: matches, rows, products,
 *    monoliths, offsets, cursor and stream batches, and attention outputs alike. Only fixed-size
 *    reports, such as stats, descriptions, metadata and the scalar @c written and @c required, stay
 *    caller fields, and so does state the caller carries between calls, like attention's @c carry.
 *  - Executors make arenas. @c usearch_arena_init places one where its executor computes, host pages
 *    on its memory domains or the memory of its GPU, decided once; a @c NULL executor makes host
 *    pages for calls on the calling thread. A GPU arena keeps the rule @ref usearch_arena_t notes.
 *  - A call rewinds the arena it is given, invalidating the previous call's outputs, and writes views
 *    into it. Any verb may reuse any arena of its residency, one call at a time.
 *  - Blocking calls borrow the arena; deferred calls hold it until final.
 *
 *  Outputs are views the call writes, so none carries a capacity or a stride for the caller to
 *  pick, and an arena keeps its high-water mark, so a steady loop allocates once. The library
 *  stages only to change a layout or to make pageable memory DMA-able. On a VRAM executor the
 *  outputs are device memory, which the caller copies out itself when it wants them elsewhere.
 *
 *  @code{.c}
 *  usearch_arena_t arena = NULL;
 *  usearch_arena_init(&(usearch_arena_init_t){.arena = &arena, .executor = executor, .outcome = &outcome});
 *  for (size_t i = 0; i != keys_count; ++i) {
 *      get.key = keys[i], get.vectors = &rows, get.written = &rows_count, get.arena = arena;
 *      usearch_dense_get(&get);
 *      if (get.outcome->status != usearch_status_success_k) break; // blocking, so already final
 *      consume(rows, rows_count); // valid until the next call given `arena`
 *  }
 *  usearch_arena_free(arena);
 *  @endcode
 *
 *  @section types_shapes Shapes
 *
 *  A verb's @c count is its batch axis and nothing else. Inputs are views, where a zero stride
 *  repeats the first element as BLAS does; outputs are views too, written packed into the arena,
 *  so none carries a capacity. Operands off the batch axis keep their own shape, and ragged inputs
 *  and outputs are tapes with Apache Arrow offsets: one more of them than entries, around the
 *  elements laid end to end.
 */
#ifndef UNUM_USEARCH_TYPES_H
#define UNUM_USEARCH_TYPES_H

#include <stddef.h> // `size_t`
#include <stdint.h> // `uint8_t`, `int32_t`, `uint32_t`

#ifdef __cplusplus
extern "C" {
#endif

/** Marks exported functions: exported while building the shared library, imported when consuming
 *  it on Windows, and of default visibility on ELF and Mach-O. Define it to override. */
#ifndef USEARCH_EXPORT
#if (defined(_WIN32) || defined(__CYGWIN__)) && defined(USEARCH_EXPORTING)
#define USEARCH_EXPORT __declspec(dllexport)
#elif defined(_WIN32) || defined(__CYGWIN__)
#define USEARCH_EXPORT __declspec(dllimport)
#elif defined(__GNUC__) || defined(__clang__)
#define USEARCH_EXPORT __attribute__((visibility("default")))
#else
#define USEARCH_EXPORT
#endif
#endif // !defined(USEARCH_EXPORT)

/** The library's version, which the release tooling rewrites in place. */
#define USEARCH_VERSION_MAJOR 3
#define USEARCH_VERSION_MINOR 0
#define USEARCH_VERSION_PATCH 0

/**
 *  @brief User-facing 128-bit key identifying one entry of any index - sixteen opaque bytes,
 *      rendered canonically as a lowercase dashed UUID, matching SafeGraphs' @c uuid key column.
 *
 *  Keys need not be unique: several entries may share one, and count-like functions report how
 *  many. Integer identities sit in the low bytes with the rest zeroed.
 */
typedef struct usearch_key_t {
    uint8_t bytes[16];
} usearch_key_t;

/** Distance or similarity score between two entries, in the metric of the index that
 *  produced it. Wide enough for every modality's scoring range. */
typedef double usearch_distance_t;

/**
 *  @brief A borrowed run of bytes and its length, never null-terminated by contract - the shape
 *      every modality's string arguments arrive in.
 *
 *  Callers holding a Rust @c str, a Go string, or a Java byte array pass their own storage without
 *  manufacturing a terminator none of them keeps.
 */
typedef struct usearch_bytes_view_t {

    /** First byte of the run; text is bytes that happen to decode. */
    uint8_t const* data;

    /** Bytes the run spans, with no terminator assumed or written. */
    size_t length;
} usearch_bytes_view_t;

/**
 *  @brief A strided run of keys - how a batch names the identities it carries, one per row of the
 *      call's @c count, or one per entry a cursor pull wrote into its arena.
 *
 *  A zero @c stride repeats the first key for every row, which is how a batch of rows lands under
 *  one key without a caller materializing the copies.
 *  An output's keys are packed, so the call writes one key's size as @c stride.
 */
typedef struct usearch_keys_view_t {

    /** First key of the run. */
    usearch_key_t const* data;

    /** Bytes between consecutive keys; zero repeats the first. */
    size_t stride;
} usearch_keys_view_t;

/**
 *  @brief What went wrong, in a vocabulary shared by the engine and by every binding that
 *      validates before calling it - so one failure carries one identity, whoever catches it.
 *
 *  Some enumerators name failures this library never returns itself: a binding holding richer
 *  layout information than the ABI can accept rejects the call before it crosses, and reports the
 *  same status the engine would if it could see the same thing. The set describes the project's
 *  failures, not this implementation's exits, so unused enumerators are deliberate.
 */
typedef enum usearch_status_t {

    /** The call succeeded; @c message carries nothing. */
    usearch_status_success_k = 0,

    /** The call was submitted deferred and is not final yet; the one non-final status. */
    usearch_status_deferred_k = 1,

    /** An allocation or reservation could not be satisfied. */
    usearch_status_out_of_memory_k = 2,

    /** Dimensionality or row counts disagree. */
    usearch_status_shape_mismatch_k = 3,

    /** A stride or packing the engine cannot traverse. */
    usearch_status_unsupported_layout_k = 4,

    /** A scalar, metric, or residency tag this build does not know. */
    usearch_status_unknown_kind_k = 5,

    /** A mutating verb on a handle that serves its checkpoint in place. */
    usearch_status_read_only_k = 6,

    /** A file could not be opened, read, written, or parsed. */
    usearch_status_io_k = 7,

    /** A device could not be selected, or its memory could not be claimed. */
    usearch_status_device_k = 8,

    /** A caller-supplied representation could not be decoded - a malformed key, a truncated
     *  buffer or checkpoint. */
    usearch_status_malformed_k = 9,

    /** The library reported neither success nor a failure - a contract violation rather
     *  than a condition, so it names a defect and not a state. */
    usearch_status_internal_k = 10,

    /** A hard design ceiling was reached, like the 32-bit ID space; a smaller request
     *  cannot help where a smaller allocation would. */
    usearch_status_overflow_k = 11,

    /** The verb needs a prior successful reservation that never happened. */
    usearch_status_uninitialized_k = 12,

    /** The request contradicts an already-frozen configuration, like re-reserving with
     *  a different geometry. */
    usearch_status_config_mismatch_k = 13,

    /** The call was refused rather than waited for, like a submission to a full deferred table. */
    usearch_status_contended_k = 14,

    /** A progress handler asked the operation to stop; it unwound and
     *  published nothing, so the handle carries whatever it did before the call. */
    usearch_status_cancelled_k = 15,

    /** The verb is declared for every residency but this one does not implement it, nor the
     *  requested traversal, or the build left the requested quantization out; no retry, option, or
     *  smaller request can help, so a caller picks another residency, traversal or build. */
    usearch_status_unsupported_k = 16,

    /** An argument breaks the verb's contract - a required pointer left @c NULL, inputs inside
     *  the arena the call rewinds, a name no table spells. */
    usearch_status_invalid_argument_k = 17,
} usearch_status_t;

/** Outcome of one call: a status every binding can branch on, and a static message
 *  carrying the detail the status cannot. Messages are static and @b don't need to be
 *  deallocated, and are @c NULL exactly when the status is @c usearch_status_success_k. */
typedef struct usearch_outcome_t {
    usearch_status_t status;
    char const* message;
} usearch_outcome_t;

/** Scalar types the indexes store and exchange, named by width and encoding.
 *  Sub-byte types pack several dimensions per byte, as their @c xN suffixes spell. */
typedef enum usearch_scalar_t {

    /** No scalar named - an unset field, or a custom metric owning its own storage. */
    usearch_scalar_unknown_k = 0,

    /** IEEE-754 binary64: ±1.80e308, epsilon 2.22e-16, subnormals to 4.94e-324,
     *  with infinities and NaNs. */
    usearch_scalar_f64_k = 1,

    /** IEEE-754 binary32: ±3.40e38, epsilon 1.19e-7, subnormals to 1.40e-45, with
     *  infinities and NaNs. */
    usearch_scalar_f32_k = 2,

    /** Truncated f32 - 8-bit exponent, 7-bit mantissa: f32's range at ±3.39e38 and
     *  epsilon 7.81e-3, with infinities and NaNs. */
    usearch_scalar_bf16_k = 3,

    /** IEEE-754 binary16: ±65504, epsilon 9.77e-4, subnormals to 5.96e-8, with
     *  infinities and NaNs. */
    usearch_scalar_f16_k = 4,

    /** FP8 with a 5-bit exponent: ±57344, epsilon 0.25, with infinities and NaNs -
     *  f16's dynamic range carried on two mantissa bits. */
    usearch_scalar_e5m2_k = 5,

    /** FP8 with a 4-bit exponent: ±448, epsilon 0.125, no infinities, and two NaN
     *  codes at 0x7F and 0xFF. */
    usearch_scalar_e4m3_k = 6,

    /** FP6 with a 3-bit exponent, byte-aligned with two padding bits: ±28, no
     *  infinities or NaNs, so every one of its 64 codes is a number. */
    usearch_scalar_e3m2_k = 7,

    /** FP6 with a 2-bit exponent, byte-aligned with two padding bits: ±7.5, no
     *  infinities or NaNs, and 72% of its codes outside [-1, 1]. */
    usearch_scalar_e2m3_k = 8,

    /** 8-bit signed integer spanning -128 to 127, every code a number. */
    usearch_scalar_i8_k = 10,

    /** Two signed nibbles per byte spanning -8 to 7, even dimensions in the low
     *  nibble and odd ones in the high. */
    usearch_scalar_i4x2_k = 11,

    /** 32-bit signed integer spanning ±2.15e9, every code a number - the Earth-fixed centimetres a
     *  geospatial cursor exports. */
    usearch_scalar_i32_k = 12,

    /** 8-bit unsigned integer spanning 0 to 255, every code a number. */
    usearch_scalar_u8_k = 20,

    /** Two unsigned nibbles per byte spanning 0 to 15, even dimensions in the low
     *  nibble and odd ones in the high. */
    usearch_scalar_u4x2_k = 21,

    /** Eight single-bit dimensions per byte - the form the Hamming & Jaccard metrics
     *  score, where only the set-bit count of a XOR or AND carries meaning. */
    usearch_scalar_u1x8_k = 22,

} usearch_scalar_t;

/**
 *  @brief A run of rows along a call's batch axis - the currency of every verb whose rows share one
 *      width, borrowed from the caller as an input and written packed into an arena as an output.
 *
 *  An input's row count is the call's @c count, and the call checks @c columns against the frozen
 *  geometry. A zero @c stride repeats the first row.
 *  An output's rows are packed: the call writes one row's bytes as @c stride, and @c columns and
 *  @c scalar as it wrote them, so neither a repeat nor a geometry check applies.
 */
typedef struct usearch_rows_view_t {

    /** Base pointer of the first row. */
    void const* data;

    /** Bytes between the starts of consecutive rows; zero repeats the first row. */
    size_t stride;

    /** Values per row; an input's must match the frozen geometry, and sub-byte rows of either kind
     *  must fill whole bytes, or they are refused. */
    size_t columns;

    /** The scalar type the rows hold, converted on the way in where it differs. */
    usearch_scalar_t scalar;
} usearch_rows_view_t;

/**
 *  @brief A borrowed strided matrix off the batch axis - an operand whose rows and columns are its
 *      own, like the value rows attention gathers from.
 *
 *  @c rows and @c columns are counts and @c stride alone is in bytes. A zero @c stride repeats the
 *  first row.
 */
typedef struct usearch_matrix_view_t {

    /** Base pointer of the first row. */
    void const* data;

    /** Rows the matrix carries. */
    size_t rows;

    /** Logical scalars per row, as the caller counts them. */
    size_t columns;

    /** Bytes between the starts of consecutive rows; zero repeats the first row. */
    size_t stride;

    /** The scalar type the rows hold. */
    usearch_scalar_t scalar;
} usearch_matrix_view_t;

/**
 *  @brief A ragged batch as Apache Arrow lays one out: elements end to end, and one more offset
 *      than entries around them, borrowed as an input or written into an arena as an output.
 *
 *  Entry @c i spans elements @c offsets[i] up to @c offsets[i+1], so the last offset closes the
 *  final entry. What an element is belongs to the modality: a byte of a sequence, a point of a
 *  shape or of a geospatial set.
 */
typedef struct usearch_tape_view_t {

    /** First element of the tape, which every offset counts from. */
    void const* data;

    /** Entries + 1 non-decreasing offsets, in elements: the call's @c count + 1 for an input, its
     *  @c written + 1 for an output. */
    size_t const* offsets;

    /** The scalar type the elements hold: @c u8 for sequences, @c f64 degrees for geospatial inputs
     *  and @c i32 Earth-fixed centimetres for their cursor, and any kind for shape points,
     *  converted on the way in where it differs from storage. */
    usearch_scalar_t scalar;
} usearch_tape_view_t;

/**
 *  @brief A ranked run a call wrote into its arena - @c count rows of @c columns packed columns,
 *      where @c count is the call's own.
 *
 *  Carries no query count of its own: the call already fixed it, and a second copy is a second
 *  thing to disagree. Row @c i holds @c counts[i] matches from @c keys[i · columns] and
 *  @c scores[i · columns] on, and the columns past them are unwritten.
 *
 *  What @c scores hold is each verb's own: a search writes distances nearest-first, whichever kind
 *  the index froze - the inner kind as @c 1 @c - @c dot - while the linear-algebra verbs write
 *  similarities, as their own docs state.
 */
typedef struct usearch_matches_view_t {

    /** First row of the keys, @c columns per row. */
    usearch_key_t const* keys;

    /** First row of the scores, positionally paired with @c keys. */
    usearch_distance_t const* scores;

    /** Matches written per row, one entry per row. */
    size_t const* counts;

    /** Columns per row the call wrote, at most its @c wanted - the "k" in "kANN". */
    size_t columns;
} usearch_matches_view_t;

/**
 *  @brief A ragged run of matches a call wrote into its arena, for verbs whose rows hold as many
 *      matches as their walks found: keys and scores end to end, and one more offset than rows.
 *
 *  Row @c i spans @c offsets[i] up to @c offsets[i+1] in both arrays, so the last offset is the
 *  total. Scores follow the same per-verb meaning as in @c usearch_matches_view_t.
 */
typedef struct usearch_matches_tape_view_t {

    /** Every row's keys, end to end. */
    usearch_key_t const* keys;

    /** Every row's scores, positionally paired with @c keys. */
    usearch_distance_t const* scores;

    /** The call's @c count + 1 non-decreasing offsets into @c keys and @c scores. */
    size_t const* offsets;
} usearch_matches_tape_view_t;

/** The CPU families USearch ships tuned variants for - dispatch reports its pick in
 *  this vocabulary, and callers may pin one. Each class bundles the compute features,
 *  wait instruction, and atomics its silicon generation guarantees. */
typedef enum usearch_cpu_class_t {
    usearch_cpu_class_unknown_k = 0,

    /** The portable floor: no SIMD assumptions, standard yields. */
    usearch_cpu_class_serial_k,

    /** Haswell-class x86: AVX2 & FMA, the 256-bit floor. */
    usearch_cpu_class_intel_haswell_k,

    /** Sierra Forest-class x86 - the E-core Xeons: AVX-VNNI and its successors without
     *  AVX-512, TPAUSE waits, CMPCCXADD conditional adds. */
    usearch_cpu_class_intel_sierra_k,

    /** Skylake-class x86: the AVX-512 foundation tier. */
    usearch_cpu_class_intel_skylake_k,

    /** Icelake-class x86: AVX-512 VNNI & VPOPCNT - the integer & bit workhorse. */
    usearch_cpu_class_intel_icelake_k,

    /** Genoa-class x86 - Zen 4 and newer: native bf16 dot products over AVX-512. */
    usearch_cpu_class_amd_genoa_k,

    /** Sapphire Rapids-class x86: the Genoa bundle plus AVX-512 FP16 and the AMX-BF16 &
     *  AMX-INT8 tiles. */
    usearch_cpu_class_intel_sapphire_k,

    /** Granite Rapids-class x86: the Sapphire Rapids bundle plus the AMX-FP16 tiles. */
    usearch_cpu_class_intel_granite_k,

    /** Diamond Rapids-class x86: the Granite Rapids bundle plus AVX10.2's FP8 conversions,
     *  TPAUSE waits, CMPCCXADD conditional adds. */
    usearch_cpu_class_intel_diamond_k,

    /** Graviton2-class Arm - Neoverse N1, the bare NEON floor; phones predating
     *  Armv8.6 land here too. */
    usearch_cpu_class_aws_g2_k,

    /** Armv8.6+ NEON: dot products, bf16 & f16 fused multiplies - Apple M2,
     *  Cortex-A710 & X2 class silicon. */
    usearch_cpu_class_apple_m2_k,

    /** Apple M4-class Arm: the Armv8.6 NEON bundle plus the SME outer-product tiles. */
    usearch_cpu_class_apple_m4_k,

    /** Graviton5-class Arm - Neoverse V3, Armv9.2: monitored waits guaranteed. */
    usearch_cpu_class_aws_g5_k,

    /** Vera-class Arm - NVIDIA's Armv9.2-generation cores with FP8 dot products. */
    usearch_cpu_class_nvidia_vera_k,

    /** Alibaba C930-class RISC-V - T-Head's XuanTie core, RVA23: vectors with Zvbb, Zawrs
     *  monitored waits, Zacas single-instruction compare-and-swap. */
    usearch_cpu_class_alibaba_c930_k,

    /** WebAssembly every engine loads: 32-bit, single-threaded, the fixed-width simd128 tier. */
    usearch_cpu_class_wasm32_v128_k,

    /** WebAssembly with relaxed SIMD, shared memory for real thread pools, and the 64-bit
     *  addressing that alone lets one module's memory pass 4 GB. */
    usearch_cpu_class_wasm64_extended_k,
} usearch_cpu_class_t;

/** The GPU families USearch ships tuned variants for, the device twin of
 *  @c usearch_cpu_class_t: one class per generation of kernel kits. */
typedef enum usearch_gpu_class_t {
    usearch_gpu_class_unknown_k = 0,

    /** Ampere-class NVIDIA and every GPU without a class of its own - A100, A10 & RTX 30. */
    usearch_gpu_class_nvidia_ampere_k,

    /** Ada-class NVIDIA - L4, L40S & RTX 40: the fp8 conversions. */
    usearch_gpu_class_nvidia_ada_k,

    /** Hopper-class NVIDIA - H100 & H200: no TMA gather, DPX instructions. */
    usearch_gpu_class_nvidia_hopper_k,

    /** Blackwell-class NVIDIA - B200, B300 & GB300: TMA gather4 scattered fetches. */
    usearch_gpu_class_nvidia_blackwell_k,

    /** Consumer Blackwell NVIDIA - RTX 50, RTX PRO 6000 & DGX Spark. */
    usearch_gpu_class_nvidia_blackwellrtx_k,

    /** Rubin-class NVIDIA - the Vera-Rubin platform's device half. */
    usearch_gpu_class_nvidia_rubin_k,

    /** CDNA3 AMD - MI300X & MI325X: gfx942, 64-wide waves, FNUZ fp8. */
    usearch_gpu_class_amd_cdna3_k,

    /** CDNA4 AMD - MI350X & MI355X: gfx950, OCP fp8. */
    usearch_gpu_class_amd_cdna4_k,

    /** CDNA5 AMD - MI400: gfx1250, 32-wide waves. */
    usearch_gpu_class_amd_cdna5_k,

    /** Apple7-family GPUs and every Apple GPU without a class of its own - M1 & M2. */
    usearch_gpu_class_apple_metal7_k,

    /** Apple9-family GPUs - M3 & M4. */
    usearch_gpu_class_apple_metal9_k,

    /** Apple10-family GPUs - M5: the Neural Accelerator matrix units. */
    usearch_gpu_class_apple_metal10_k,
} usearch_gpu_class_t;

/** The GPU vendors one binary drives at once, each through a runtime loaded on first use. */
typedef enum usearch_gpu_vendor_t {
    usearch_gpu_vendor_unknown_k = 0,

    /** NVIDIA, through the CUDA driver when one is installed. */
    usearch_gpu_vendor_nvidia_k,

    /** AMD, through the HIP runtime when one is installed. */
    usearch_gpu_vendor_amd_k,

    /** Apple, through Metal, present on every Apple silicon Mac. */
    usearch_gpu_vendor_apple_k,
} usearch_gpu_vendor_t;

/** One GPU as USearch numbers it: its position in @c usearch_gpu_devices is the USearch device
 *  ordinal every VRAM call names. */
typedef struct usearch_gpu_device_t {

    /** The vendor whose runtime drives the device. */
    usearch_gpu_vendor_t vendor;

    /** The device's ordinal in its vendor's runtime, as @c nvidia-smi or @c rocm-smi list it. */
    uint32_t vendor_ordinal;

    /** The class whose kernels run on it. */
    usearch_gpu_class_t gpu_class;

    /** Streaming multiprocessors, compute units or GPU cores. */
    uint32_t compute_units;

    /** Device memory in bytes, or the working set Metal recommends for unified memory. */
    uint64_t memory_bytes;

    /** The vendor's own name for the device, NUL-terminated and truncated to fit. */
    char name[64];
} usearch_gpu_device_t;

/**
 *  @brief Where an index's payload lives and which engine traverses it - the vocabulary the
 *      lifecycle verbs spell in their names, as @c usearch_dense_init_vram does.
 *
 *  Each modality names a verb per action and residency rather than taking this as a tag, so a
 *  residency a modality lacks has no verb to call. The enumerators remain for the name tables
 *  bindings parse user text with.
 */
typedef enum usearch_residency_t {

    /** Volatile-memory index traversed by CPU threads - the concurrent baseline. */
    usearch_residency_ram_k = 0,

    /** Accelerator-memory index traversed by GPU warps on a chosen device. */
    usearch_residency_vram_k = 1,

    /** Persisted index traversed through pluggable I/O engines, sized past RAM. */
    usearch_residency_nvme_k = 2,
} usearch_residency_t;

/** How a search reaches the indexed rows it scores. */
typedef enum usearch_traversal_t {

    /** Walk the graph, scoring the rows the expansion reaches - approximate, the default. */
    usearch_traversal_walk_k = 0,

    /** Score every row published when the query's scan starts - exact, the ground truth a walk
     *  approximates. */
    usearch_traversal_scan_k = 1,
} usearch_traversal_t;

/** Traversal telemetry, shared by every verb that walks a graph in every modality - a pure output,
 *  fully overwritten by the call it is passed to. Passing @c NULL skips reporting. */
typedef struct usearch_traversal_stats_t {

    /** Graph members visited while resolving the batch. */
    size_t visited_members;

    /** Distances computed while resolving the batch, diversity re-selections included. */
    size_t computed_distances;

    /** Starvation restarts - re-seedings after the frontier drained prematurely,
     *  and a graph-connectivity health signal rather than a cost counter. */
    size_t restarts;
} usearch_traversal_stats_t;

/** One reachability audit: the population it judged, and how much of it a search reaches. */
typedef struct usearch_reachability_t {

    /** Entries published when the audit started - the population it judges; entries inserted
     *  while it ran are not counted, so compare @c reachable against this, never a later size. */
    size_t audited;

    /** Of those, the entries a search reaches from the landmarks; the rest no search finds until
     *  a refinement re-links them. */
    size_t reachable;

    /** Levels the sweep took to converge - a diameter proxy; the deepest shard's on a sharded
     *  handle. */
    size_t sweeps;
} usearch_reachability_t;

/**
 *  @brief The host memory domains a RAM-resident index places its shards in - one shard per
 *      named NUMA node, so a dual-socket box names two and gets two graphs whose traversals
 *      stay local to the cores that walk them.
 *
 *  Names where the index lives, never who walks it: threads belong to the executor the verbs
 *  take. Every modality's RAM lifecycle verbs and the RAM executor take this struct.
 *
 *  The library's own payload blocks always claim huge pages, falling back to base pages where no
 *  pool is provisioned.
 */
typedef struct usearch_ram_options_t {

    /** NUMA node IDs to place shards on, borrowed for the duration of the call;
     *  @c NULL derives every node the process may allocate on. */
    int32_t const* memory_domains;

    /** Number of entries in @c memory_domains; zero derives. */
    size_t memory_domains_count;
} usearch_ram_options_t;

/**
 *  @brief The devices a VRAM-resident index places its shards on - one shard per named ordinal,
 *      sharded and never replicated, so eight devices host eight slices of one graph rather than
 *      eight copies of it.
 *
 *  Names where the index lives, never who walks it: streams belong to the executor the verbs
 *  take, one per device. A checkpoint of more parts than named devices keeps one shard per part,
 *  part @c p on device @c p modulo their count, and shards sharing a device queue on its stream
 *  in order.
 *
 *  Flat matrices are device-resident, never migratable: an index outgrowing the named devices
 *  fails rather than paging to the host, and the answer is to name more devices.
 *
 *  One index spans one vendor and one class, so naming devices of two vendors or two generations
 *  refuses with @c usearch_status_config_mismatch_k; separate indexes may sit on different vendors
 *  in one process.
 */
typedef struct usearch_vram_options_t {

    /** USearch device ordinals to place shards on, NVIDIA devices first, then AMD, then Apple, as
     *  @c usearch_gpu_devices lists them, borrowed for the call; @c NULL picks device zero. */
    int32_t const* devices;

    /** Number of entries in @c devices; zero derives. */
    size_t devices_count;
} usearch_vram_options_t;

/**
 *  @brief The devices an NVMe-resident index serves its shards from - one shard per named path,
 *      so striping across four drives names four.
 *
 *  Names where the index lives, never who walks it: submission depth belongs to the executor the
 *  verbs take. Naming nothing serves every shard where its manifest already points.
 *
 *  A path is whatever identifies a device to the engine that can reach it, so the string is
 *  the extension point rather than a field per transport. A filesystem path names a kernel
 *  device; a transport identifier like @c trtype:PCIe @c traddr:0000:5e:00.0 names a
 *  namespace the kernel no longer owns and selects a userspace engine. A FUSE mount is just
 *  a path too, which is how object storage stages in without an HTTP stack in this library -
 *  point @c mountpoint-s3 at a bucket and the ordinary engines read it.
 *
 *  Transfer geometry is never an input: alignment is probed from the device, and the record
 *  granularity a checkpoint was written at is read back from its own header.
 *
 *  Reads always bypass the page cache where the filesystem admits @c O_DIRECT.
 */
typedef struct usearch_nvme_options_t {

    /** Devices or directories to serve shards from, borrowed for the duration of the
     *  call; @c NULL leaves every shard where its manifest points. */
    usearch_bytes_view_t const* paths;

    /** Number of entries in @c paths; zero derives. */
    size_t paths_count;
} usearch_nvme_options_t;

/** Whether a filtered search admits one candidate key or passes over it. */
typedef enum usearch_admission_t {

    /** Skip the key; the walk still routes through it. */
    usearch_reject_k = 0,

    /** Report the key among the matches. */
    usearch_admit_k = 1,
} usearch_admission_t;

/** Admits or rejects @p count candidate keys at once, never zero, writing one admission per key
 *  into @p admissions, a buffer the library owns for the call, so a handler backed by an external
 *  store answers them in one round trip. Called concurrently from the executor's threads. */
typedef void (*usearch_predicate_callback_t)(usearch_key_t const* keys, usearch_admission_t* admissions, size_t count,
                                             void* state);

/** Which candidate keys a filtered search may report: a handler and the state it is called with. */
typedef struct usearch_predicate_t {

    /** The @b optional handler; @c NULL admits every key. */
    usearch_predicate_callback_t callback;

    /** Opaque pointer forwarded into every @c callback call. */
    void* state;
} usearch_predicate_t;

/** Whether the operation a handler is watching carries on or unwinds. */
typedef enum usearch_continuation_t {

    /** Carry on to the next unit of work. */
    usearch_continue_k = 0,

    /** Stop and report @c usearch_status_cancelled_k; whether finished increments stand
     *  is each verb's own contract. */
    usearch_cancel_k = 1,
} usearch_continuation_t;

/**
 *  @brief Periodic report of a long operation's advance, and the caller's chance to stop it.
 *      Counts are in whatever units the verb consumes - rows for an insertion, queries for a
 *      search, shards for a checkpoint - so one handler serves every one of them.
 *
 *  Never invoked concurrently, however many threads perform the work, so the handler needs no
 *  synchronization of its own - and never after the call it was passed to stores its final status.
 *
 *  Arguments, in order:
 *  - @c completed: units finished so far, never decreasing across one call's reports.
 *  - @c total: units the call finishes unless cancelled.
 *  - @c state: the opaque pointer registered alongside the handler, forwarded verbatim.
 */
typedef usearch_continuation_t (*usearch_progress_callback_t)( //
    size_t completed, size_t total, void* state);

/**
 *  @brief How a long call reports its advance: a handler, its state, and how many reports it hears.
 *
 *  A batch verb splits its units into @c steps nearly equal slices, hands each to its engine whole
 *  and reports after each; a verb whose engine counts its own units, a checkpoint's shards or a
 *  refinement's nodes, forwards only the ticks that cross into the next slice. Without a handler
 *  the call runs as one slice and reports nothing.
 */
typedef struct usearch_progress_t {

    /** The @b optional handler; @c NULL skips every report. */
    usearch_progress_callback_t callback;

    /** Opaque pointer forwarded into every @c callback call. */
    void* state;

    /** Reports a call makes at most, the last at completion; zero derives to ten. */
    size_t steps;
} usearch_progress_t;

/**
 *  @brief Blocks on an atomic wait until @p outcome is final, with acquire ordering.
 *  @param[in] outcome An outcome passed to a call, read until its status leaves @c usearch_status_deferred_k.
 *  @return The final status.
 *
 *  Never wait on a call's own outcome from inside its progress handler or predicate: the call
 *  finishes only after they return.
 */
USEARCH_EXPORT usearch_status_t usearch_outcome_wait(usearch_outcome_t* outcome);

/**
 *  @brief Reads the status of @p outcome without waiting, with the acquire ordering its final
 *      store pairs with, so a final status also publishes every output of the call.
 *  @param[in] outcome An outcome passed to a call.
 *  @return @c usearch_status_deferred_k while the call is not final, then its final status.
 *
 *  The one way to poll: the status word is atomic only by this contract, so a caller reading the
 *  field directly needs an atomic load its language may not offer on foreign memory.
 */
USEARCH_EXPORT usearch_status_t usearch_outcome_status(usearch_outcome_t* outcome);

/** Whether a call returns with its final status, or once submitted to its executor. */
typedef enum usearch_completion_t {

    /** Return with the final status stored and every output readable; the default. */
    usearch_completion_blocking_k = 0,

    /** Return once submitted, the status deferred until final; the caller waits or polls. */
    usearch_completion_deferred_k = 1,
} usearch_completion_t;

/**
 *  @brief Everything a call needs to run but an index does not own: the host threads or device
 *      streams that advance the work, the traversal scratch each of them walks with, and the
 *      pinned bounce buffer staging passes through.
 *
 *  One executor serves any number of index handles, so separate categories of indexes share one
 *  machine's cores or devices instead of each claiming a private set. Its kind is fixed by the
 *  constructor that produced it and must match the residency of every handle it drives; a mismatch
 *  reports @c usearch_status_device_k.
 *
 *  Scratch is carved one slice per thread or stream, sized from geometry rather than
 *  from the scalar type, which is what lets a single executor drive indexes of different
 *  quantizations and metrics. It grows to the widest geometry asked of it, is rewound per call,
 *  and is kept between calls, so a steady workload allocates once.
 *
 *  Calls on one executor start in submission order, and handles stay freely concurrent - it is the
 *  executor that orders, never the index. A blocking call runs alone: after every earlier call
 *  finishes, and before any later one starts. An executor without threads of its own -
 *  caller-inclusive, single-thread, or WebAssembly - runs a deferred call at submission or holds it
 *  until its wait, its executor's next drain or blocking call, or the free of a handle it names.
 *
 *  Every executor tracks its deferred calls in a fixed table of @c deferred_limit slots, so
 *  submitting never allocates. A deferred submission that finds the table full finishes
 *  @c usearch_status_contended_k, and blocking calls take no slot.
 *
 *  A blocking verb or @c usearch_executor_drain called from inside a progress handler or a predicate
 *  of a call on the same executor reports @c usearch_status_contended_k rather than waiting on itself.
 *
 *  Every verb accepts a @c NULL executor and then runs on the caller's thread, blocking whatever
 *  its @c completion says: RAM work runs single-threaded, VRAM work runs on the handle's device and
 *  default stream and synchronizes before returning, and NVMe reads are synchronous. Its outputs
 *  land in any arena of the handle's residency.
 *
 *  The library stages only to change layout on the side that computes, or to make pageable memory
 *  DMA-able, and never routes device bytes through the host:
 *
 *      Case                                          Action
 *      Device inputs on the executor's device        read in place
 *      Device inputs needing padding or a cast       a device-to-device copy or a cast kernel
 *      Pinned or managed host inputs                 a direct asynchronous upload
 *      Pageable host inputs                          the pinned bounce buffer, then an upload
 *      Outputs in an arena on the executor's device  the kernel writes them directly
 */
typedef struct usearch_executor_handle_t* usearch_executor_t;

/** The arguments of @c usearch_executor_for_ram. */
typedef struct usearch_executor_for_ram_t {

    /** Output for the new executor. */
    usearch_executor_t* executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** The memory domains of the indexes it will drive; copied, not borrowed. */
    usearch_ram_options_t ram;

    /** Threads to spawn; zero takes 7/8 of the cores those domains carry. */
    size_t threads;

    /** Deferred calls it tracks at once, sizing its deferred table; zero takes a default. */
    size_t deferred_limit;
} usearch_executor_for_ram_t;

/**
 *  @brief Spawns host threads for RAM-resident indexes, interleaved across the compute domains
 *      local to the named memory domains, or borrows an existing ForkUnion pool.
 *
 *  Passing the very placement the indexes were built with is what makes compute and memory domains
 *  agree, leaving no second place to name hardware and no way for the two to disagree.
 *
 *  Interleaving is why a half-size executor takes half of every socket rather than all of one, and
 *  the default stops at 7/8 of the cores because full occupancy starves the ordered publication
 *  path. A borrowed pool keeps its own threads and its lender's mood, so one machine runs one pool
 *  however many libraries schedule onto it.
 */
USEARCH_EXPORT void usearch_executor_for_ram(usearch_executor_for_ram_t* arguments);

/** The arguments of @c usearch_executor_on_pool. */
typedef struct usearch_executor_on_pool_t {

    /** Output for the new executor. */
    usearch_executor_t* executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** The memory domains of the indexes it will drive; copied, not borrowed. */
    usearch_ram_options_t ram;

    /** The ForkUnion @c fu_pool_t to borrow; it must outlive the executor, and keeps its lender's mood. */
    void* pool;

    /** Deferred calls it tracks at once, sizing its deferred table; zero takes a default. */
    size_t deferred_limit;
} usearch_executor_on_pool_t;

/**
 *  @brief Creates a RAM executor over a borrowed ForkUnion pool, so several libraries in one process share one set
 *      of threads while each keeps its own scratch.
 *  @param[in] arguments Named arguments, whose outputs are written through their pointers.
 */
USEARCH_EXPORT void usearch_executor_on_pool(usearch_executor_on_pool_t* arguments);

/** The arguments of @c usearch_executor_for_vram. */
typedef struct usearch_executor_for_vram_t {

    /** Output for the new executor. */
    usearch_executor_t* executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** The devices of the indexes it will drive; copied, not borrowed. */
    usearch_vram_options_t vram;

    /** The @b optional borrowed CUDA streams, one per device in @c vram order; @c NULL creates its
     *  own. */
    void* const* streams;

    /** The @b optional ceiling on scratch bytes per stream, a refinement's included, past which a
     *  call reports @c usearch_status_out_of_memory_k; zero lets scratch grow as geometry asks. */
    size_t scratch_limit;

    /** Deferred calls it tracks at once, sizing its deferred table; zero takes a default. */
    size_t deferred_limit;
} usearch_executor_for_vram_t;

/**
 *  @brief Opens one submission stream on every distinct device @c vram names, or borrows the
 *      caller's, all of them driven by whichever host thread submits.
 *
 *  One stream per device is the whole design: every shard on a device queues on its stream in
 *  order, a single host thread keeps every device fed, and adding host threads would buy nothing,
 *  since the work has already left for the accelerator by the time one of them would return. A
 *  GPU executor never changes process-wide device flags.
 */
USEARCH_EXPORT void usearch_executor_for_vram(usearch_executor_for_vram_t* arguments);

/** The arguments of @c usearch_gpu_devices. */
typedef struct usearch_gpu_devices_t {

    /** Output for the outcome, final when the call returns. */
    usearch_outcome_t* outcome;

    /** Output for up to @c devices_capacity devices, in USearch ordinal order; may be @c NULL. */
    usearch_gpu_device_t* devices;

    /** Entries @c devices holds. */
    size_t devices_capacity;

    /** Output for every device this process can drive, which may exceed @c devices_capacity. */
    size_t* devices_count;
} usearch_gpu_devices_t;

/**
 *  @brief Lists every GPU this process can drive, NVIDIA devices first, then AMD, then Apple, so
 *      the position of a device is the USearch ordinal @c usearch_vram_options_t names it by.
 *
 *  Each vendor's runtime loads on first use, and a vendor whose runtime is absent lists no devices,
 *  so the call succeeds with a zero count on a machine without GPUs. The listing is taken once per
 *  process and never changes after.
 */
USEARCH_EXPORT void usearch_gpu_devices(usearch_gpu_devices_t* arguments);

/** The arguments of @c usearch_executor_for_nvme. */
typedef struct usearch_executor_for_nvme_t {

    /** Output for the new executor. */
    usearch_executor_t* executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** The devices of the indexes it will drive; copied, not borrowed. */
    usearch_nvme_options_t nvme;

    /** Threads to spawn, one submission ring apiece; zero derives from the devices' queue capabilities. */
    size_t threads;

    /** Deferred calls it tracks at once, sizing its deferred table; zero takes a default. */
    size_t deferred_limit;
} usearch_executor_for_nvme_t;

/**
 *  @brief Spawns host threads for NVMe-resident indexes, one submission ring apiece, interleaved
 *      across the compute domains local to the devices the paths name.
 *
 *  One ring per thread is what keeps submission lock-free: a thread owns its ring outright, so
 *  an external-memory walk never coordinates with a sibling to issue a read. A ring reports its
 *  reads only to the thread that owns it, so deferred calls finish on those threads.
 */
USEARCH_EXPORT void usearch_executor_for_nvme(usearch_executor_for_nvme_t* arguments);

/** What an executor does with its threads or streams once it goes idle. */
typedef enum usearch_mood_t {

    /** Idle threads sleep; the next call pays a wake-up. */
    usearch_mood_parked_k = 0,

    /** Idle threads stay hot, trading power for the next call's latency. */
    usearch_mood_spinning_k = 1,
} usearch_mood_t;

/** The arguments of @c usearch_executor_drain. */
typedef struct usearch_executor_drain_t {

    /** The executor whose submitted calls are awaited. */
    usearch_executor_t executor;

    /** Output for an executor-level failure, such as a lost device; never one of a drained call. */
    usearch_outcome_t* outcome;

    /** What the executor does whenever it next goes idle; the last drain wins. */
    usearch_mood_t mood;
} usearch_executor_drain_t;

/**
 *  @brief Waits for every call submitted to the executor before the drain began, then sets the
 *      mood it adopts whenever it next goes idle.
 *
 *  The drain reads the submission ticket on entry and waits up to it, so calls submitted while it
 *  waits can't keep it waiting forever. Every call reports its own failure through its own
 *  @c outcome, so a drain reports only what befell the executor itself.
 *
 *  Parked means futex waits on RAM, blocking-sync events on VRAM, and ring waits on NVMe; spinning
 *  polls each of them instead. A borrowed pool keeps its lender's mood.
 */
USEARCH_EXPORT void usearch_executor_drain(usearch_executor_drain_t* arguments);

/**
 *  @brief Drains every call, releases the scratch and the streams it owns, and frees the executor;
 *      @c NULL is safe, and it cannot fail, so takes no outcome.
 *  @param[inout] executor The handle to be freed; invalid afterwards.
 *
 *  Waits until every deferred call naming the handle is final.
 *  Waits until every call running on the executor, blocking or deferred, is final.
 *  Never call it from inside a progress handler, a predicate, or a host function the executor
 *  runs, since it would drain itself. Host arenas it made stay valid, each keeping its memory
 *  until freed; GPU arenas keep the rule @ref usearch_arena_t notes.
 */
USEARCH_EXPORT void usearch_executor_free(usearch_executor_t executor);

/**
 *  @brief Memory the library writes every output whose length scales with a call into, reused
 *      across calls. @ref types_memory
 *
 *  Made by @c usearch_arena_init where its executor computes: host pages on its memory domains, or
 *  the memory of its GPU, and a GPU that runs out reports @c usearch_status_out_of_memory_k rather
 *  than falling back to the host. Each call given the arena rewinds it, which invalidates every
 *  output of the previous call, and refills it, growing only when the new outputs outsize
 *  everything it held, so a steady loop allocates once. To keep one call's outputs alive past the
 *  next call, pass the next call another arena.
 *
 *  - A @c NULL arena is refused with @c usearch_status_invalid_argument_k.
 *  - A call whose inputs point into the arena it is about to rewind reports
 *    @c usearch_status_invalid_argument_k.
 *  - An arena serves one call at a time; a second call naming it finishes
 *    @c usearch_status_contended_k, and a deferred one does so at submission.
 *  - A call on another residency or device than the arena's, or a call naming a GPU arena with
 *    another executor than the one that made it, @c NULL included, reports
 *    @c usearch_status_device_k.
 *  - A blocking call borrows the arena until it returns, and a deferred one holds it until final.
 *  - A call that fails, is cancelled, or has nothing to report leaves its outputs @c NULL.
 *
 *  @note A GPU arena grows, fills and frees on the first stream of the executor that made it, so it
 *      serves only calls on that executor, and must be freed before that executor is.
 */
typedef struct usearch_arena_handle_t* usearch_arena_t;

/** The arguments of @c usearch_arena_init. */
typedef struct usearch_arena_init_t {

    /** Output for the new arena. */
    usearch_arena_t* arena;

    /** The executor whose calls the arena serves, which decides where it lives; @c NULL serves calls
     *  on the calling thread from host pages. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;
} usearch_arena_init_t;

/**
 *  @brief Creates an empty arena where @c executor computes, so its placement is decided once and
 *      GPU code keeps its outputs on the device without asking.
 *
 *  Nothing is allocated until a call writes into it. An arena of host pages may outlive the
 *  executor that made it, keeping its own memory until it is freed; a GPU arena keeps the rule
 *  @ref usearch_arena_t notes.
 */
USEARCH_EXPORT void usearch_arena_init(usearch_arena_init_t* arguments);

/**
 *  @brief Frees the arena and every output in it; @c NULL is safe, and it cannot fail, so takes no
 *      outcome.
 *  @param[inout] arena The arena to be freed; invalid afterwards.
 *
 *  Waits until every deferred call naming the handle is final.
 *  Free a GPU arena once the caller's own device work on its outputs has finished, keeping the
 *  rule @ref usearch_arena_t notes.
 */
USEARCH_EXPORT void usearch_arena_free(usearch_arena_t arena);

/**
 *  @brief The canonical lowercase, dash-separated spelling of each closed set's enumerators,
 *      and the inverse that reads one back.
 *
 *  Every binding that touches user text otherwise writes these tables itself, against a header
 *  that nothing checks it against. The names are static storage and are never deallocated.
 *  A value no status carries spells "unrecognized".
 */
USEARCH_EXPORT char const* usearch_status_name(usearch_status_t status);

/** Reads a status back; @c usearch_status_invalid_argument_k when unrecognized. */
USEARCH_EXPORT usearch_status_t usearch_status_from_name(usearch_bytes_view_t name);

/** The canonical spelling of a scalar kind, as @c usearch_scalar_f32_k spells "f32". */
USEARCH_EXPORT char const* usearch_scalar_name(usearch_scalar_t scalar);

/** Reads a scalar kind back; @c usearch_scalar_unknown_k when unrecognized. */
USEARCH_EXPORT usearch_scalar_t usearch_scalar_from_name(usearch_bytes_view_t name);

/** The canonical spelling of a residency, as @c usearch_residency_ram_k spells "ram". */
USEARCH_EXPORT char const* usearch_residency_name(usearch_residency_t residency);

/** Reads a residency back; @c usearch_residency_ram_k when unrecognized. */
USEARCH_EXPORT usearch_residency_t usearch_residency_from_name(usearch_bytes_view_t name);

/** The canonical spelling of a traversal, as @c usearch_traversal_scan_k spells "scan". */
USEARCH_EXPORT char const* usearch_traversal_name(usearch_traversal_t traversal);

/** Reads a traversal back; @c usearch_traversal_walk_k when unrecognized. */
USEARCH_EXPORT usearch_traversal_t usearch_traversal_from_name(usearch_bytes_view_t name);

/** The canonical spelling of a mood, as @c usearch_mood_parked_k spells "parked". */
USEARCH_EXPORT char const* usearch_mood_name(usearch_mood_t mood);

/** Reads a mood back; @c usearch_mood_parked_k when unrecognized. */
USEARCH_EXPORT usearch_mood_t usearch_mood_from_name(usearch_bytes_view_t name);

/** The canonical spelling of a CPU class, as the dispatch ladder names its rungs. */
USEARCH_EXPORT char const* usearch_cpu_class_name(usearch_cpu_class_t cpu_class);

/** The canonical spelling of a GPU class, as the dispatch ladder names its rungs. */
USEARCH_EXPORT char const* usearch_gpu_class_name(usearch_gpu_class_t gpu_class);

/**
 *  @brief Reports the version of the library this process linked, which may differ from the
 *      @c USEARCH_VERSION_MAJOR family of macros it was compiled against.
 *  @param[out] major The major version, which changes whenever an argument structure does.
 *  @param[out] minor The minor version.
 *  @param[out] patch The patch version.
 */
USEARCH_EXPORT void usearch_version(int* major, int* minor, int* patch);

/** The version of the library as a static null-terminated string, like "3.0.0". */
USEARCH_EXPORT char const* usearch_version_name(void);

/**
 *  @brief Retrieves the hardware capabilities this binary was compiled with.
 *  @return A comma-separated static string with capability names.
 */
USEARCH_EXPORT char const* usearch_capabilities_compiled(void);

/**
 *  @brief Retrieves the hardware capabilities the current machine offers at runtime.
 *  @return A comma-separated static string with capability names.
 */
USEARCH_EXPORT char const* usearch_capabilities_available(void);

#ifdef __cplusplus
}
#endif

#endif // UNUM_USEARCH_TYPES_H
