/**
 *  @file include/usearch/types.h
 *  @author Ash Vardanian
 *  @date August 14, 2026
 *  @brief Shared vocabulary of the USearch C interface: scalar types, keys, distances, error
 *      reporting, residency selection, and library introspection.
 *
 *  Everything the modality headers have in common - @c dense_index.h, @c sequence_index.h, and
 *  @c geospatial_index.h - and nothing they don't. Include a modality header directly, since it
 *  pulls this one in; there is no umbrella header. The C vocabulary sits beside its C++ twin
 *  @c types.hpp, so one installed tree serves both languages and the headers pair by stem.
 */
#ifndef UNUM_USEARCH_TYPES_H
#define UNUM_USEARCH_TYPES_H

#include <stdbool.h> // `bool`
#include <stddef.h>  // `size_t`
#include <stdint.h>  // `uint64_t`

#ifdef __cplusplus
extern "C" {
#endif

#ifndef USEARCH_EXPORT
#if defined(_WIN32) && !defined(__MINGW32__)
#define USEARCH_EXPORT __declspec(dllexport)
#else
#define USEARCH_EXPORT
#endif
#endif

/** The library's version, which the release tooling rewrites in place. */
#define USEARCH_VERSION_MAJOR 3
#define USEARCH_VERSION_MINOR 0
#define USEARCH_VERSION_PATCH 0

/**
 *  @brief User-facing 128-bit key identifying one entry of any index - sixteen opaque bytes,
 *      rendered canonically as a lowercase dashed UUID, matching SafeGraphs' @c uuid key column.
 *
 *  Keys need not be unique: several entries may share one, and count-like functions report how
 *  many. Integer identities park in the low bytes with the rest zeroed.
 */
USEARCH_EXPORT typedef struct usearch_key_t {
    uint8_t bytes[16];
} usearch_key_t;

/** Distance or similarity score between two entries, in the metric of the index that
 *  produced it. Wide enough for every modality's scoring range. */
USEARCH_EXPORT typedef double usearch_distance_t;

/**
 *  @brief A borrowed run of bytes and its length, never null-terminated by contract - the shape
 *      every modality's string arguments arrive in.
 *
 *  Callers holding a Rust @c str, a Go string, or a Java byte array pass their own storage without
 *  manufacturing a terminator none of them keeps.
 */
USEARCH_EXPORT typedef struct usearch_bytes_view_t {

    /** First byte of the run; text is bytes that happen to decode. */
    uint8_t const* data;

    /** Bytes the run spans, with no terminator assumed or written. */
    size_t length;
} usearch_bytes_view_t;

/**
 *  @brief A caller-owned, writable run of bytes - the mutable twin of @c usearch_bytes_view_t,
 *      carried by every verb that fills a buffer the caller allocated.
 *
 *  Pairing the pointer with its length in one argument is what makes a mismatched capacity
 *  unrepresentable, where two loose arguments only made it unlikely.
 */
USEARCH_EXPORT typedef struct usearch_bytes_span_t {

    /** First byte of the run; text is bytes that happen to decode. */
    uint8_t* data;

    /** Bytes the run can hold, with no terminator assumed or written. */
    size_t length;
} usearch_bytes_span_t;

/**
 *  @brief A borrowed run of keys and its length - how a batch names the identities it carries.
 *
 *  One key per row of the matrix it accompanies, so the two counts are checked against each other
 *  rather than assumed equal, which is the check every binding otherwise writes for itself.
 */
USEARCH_EXPORT typedef struct usearch_keys_view_t {

    /** First key of the run. */
    usearch_key_t const* data;

    /** Keys the run spans. */
    size_t count;
} usearch_keys_view_t;

/**
 *  @brief A caller-owned, writable run of keys - the mutable twin of @c usearch_keys_view_t,
 *      carried by every verb that fills a key array the caller allocated.
 *
 *  Its capacity is the only bound a streaming verb has on how much it may write, so the pointer
 *  and that bound travel as one argument rather than as two a caller could contradict.
 */
USEARCH_EXPORT typedef struct usearch_keys_span_t {

    /** First key of the run. */
    usearch_key_t* data;

    /** Keys the run can hold. */
    size_t capacity;
} usearch_keys_span_t;

/**
 *  @brief What went wrong, in a vocabulary shared by the engine and by every binding that
 *      validates before calling it - so one failure carries one identity, whoever catches it.
 *
 *  Some enumerators name failures this library never returns itself: a binding holding richer
 *  layout information than the ABI can accept rejects the call before it crosses, and reports the
 *  same status the engine would if it could see the same thing. The set describes the project's
 *  failures, not this implementation's exits, so unused enumerators are deliberate.
 */
USEARCH_EXPORT typedef enum usearch_status_t {

    /** The call completed; @c message carries nothing. */
    usearch_success_k = 0,

    /** An allocation or reservation could not be satisfied. */
    usearch_error_out_of_memory_k = 1,

    /** Dimensionality, row counts, or output capacity disagree. */
    usearch_error_shape_mismatch_k = 2,

    /** A stride or packing the engine cannot traverse. */
    usearch_error_unsupported_layout_k = 3,

    /** A scalar, metric, or residency tag this build does not know. */
    usearch_error_unknown_kind_k = 4,

    /** A mutating verb on a handle that opens its checkpoint in place. */
    usearch_error_read_only_k = 5,

    /** A file could not be opened, read, written, or parsed. */
    usearch_error_io_k = 6,

    /** A device could not be selected, or its memory could not be claimed. */
    usearch_error_device_k = 7,

    /** A caller-supplied representation could not be decoded - a malformed key, a truncated
     *  buffer, a checkpoint that ends early. */
    usearch_error_malformed_input_k = 8,

    /** The library reported neither success nor a failure - a contract violation rather
     *  than a condition, so it names a defect and not a state. */
    usearch_error_internal_k = 9,

    /** A hard design ceiling was reached, like the 32-bit id space; a smaller request
     *  cannot help where a smaller allocation would. */
    usearch_error_overflow_k = 10,

    /** The verb needs a prior successful reservation that never happened. */
    usearch_error_uninitialized_k = 11,

    /** The request contradicts an already-frozen configuration, like re-reserving with
     *  a different geometry. */
    usearch_error_config_mismatch_k = 12,

    /** A synchronization contract was violated by a concurrent operation; discard any
     *  produced output and retry in a quieter phase. */
    usearch_error_contended_k = 13,

    /** A progress handler asked the operation to stop; it unwound and published nothing,
     *  so the handle carries whatever it did before the call. */
    usearch_error_cancelled_k = 14,
    /** @brief The verb is declared for every residency but this one does not implement it; no
     *      retry, option, or smaller request can help, so a caller picks another residency. */
    usearch_error_unsupported_k = 15,
} usearch_status_t;

/** Outcome of one call: a status every binding can branch on, and a static message
 *  carrying the detail the status cannot. Messages are static and @b don't need to be
 *  deallocated, and are @c NULL exactly when the status is @c usearch_success_k. */
USEARCH_EXPORT typedef struct usearch_error_t {
    usearch_status_t status;
    char const* message;
} usearch_error_t;

/** Scalar types the indexes store and exchange, named by width and encoding.
 *  Sub-byte types pack several dimensions per byte, as their @c xN suffixes spell. */
USEARCH_EXPORT typedef enum usearch_scalar_t {

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
 *  @brief A borrowed strided matrix - the batch currency every modality's row inputs take.
 *
 *  One row is a one-row matrix, so a caller never packs a special case. The row count and
 *  the column count are scalars; @c stride_bytes alone is a byte count, which is why it says
 *  so in its name. A zero @c stride_bytes means rows sit contiguously at their natural width.
 *
 *  @c columns is what the engine checks the frozen geometry against, so a matrix whose width
 *  disagrees with the index is refused with @c usearch_error_shape_mismatch_k rather than read
 *  past its end.
 */
USEARCH_EXPORT typedef struct usearch_matrix_view_t {

    /** Base pointer of the first row. */
    void const* data;

    /** Rows the matrix carries. */
    size_t rows;

    /** Logical scalars per row, as the caller counts them. */
    size_t columns;

    /** Bytes between the starts of consecutive rows; zero means contiguous. */
    size_t stride_bytes;

    /** The scalar type the rows hold, converted on the way in where it differs. */
    usearch_scalar_t scalar_kind;
} usearch_matrix_view_t;

/** The writable twin of @c usearch_matrix_view_t, carried by every verb that
 *  lands rows in a matrix the caller allocated. */
USEARCH_EXPORT typedef struct usearch_matrix_span_t {

    /** Base pointer of the first row. */
    void* data;

    /** Rows the matrix can hold, which a verb writes up to and reports back. */
    size_t capacity;

    /** Logical scalars per row, as the caller counts them. */
    size_t columns;

    /** Bytes between the starts of consecutive rows; zero means contiguous. */
    size_t stride_bytes;

    /** The scalar type to write, converted on the way out where it differs. */
    usearch_scalar_t scalar_kind;
} usearch_matrix_span_t;

/**
 *  @brief Where a ranked run lands - three caller-owned outputs sharing the queries' count.
 *
 *  Carries no query count of its own: the queries the call took already fixed it, and a second
 *  copy is a second thing to disagree. The strides stay because a caller may land results inside
 *  a wider host matrix, which @c wanted alone cannot express.
 *
 *  @c scores are nearest-first, whichever kind the index froze - the inner kind reaches the
 *  ranking as @c 1 @c - @c dot, like every other distance.
 */
USEARCH_EXPORT typedef struct usearch_matches_span_t {

    /** First row of the key output. */
    usearch_key_t* keys;

    /** Bytes between the starts of consecutive key rows; zero packs them. */
    size_t keys_stride_bytes;

    /** First row of the score output, positionally paired with @c keys. */
    usearch_distance_t* scores;

    /** Bytes between the starts of consecutive score rows; zero packs them. */
    size_t scores_stride_bytes;

    /** Matches written per query - one entry per query, never strided. */
    size_t* counts;

    /** Columns per row, and the upper bound on matches - the "k" in "kANN". */
    size_t wanted;
} usearch_matches_span_t;

/** The CPU families USearch ships tuned variants for - dispatch reports its pick in
 *  this vocabulary, and callers may pin one. Each class bundles the compute features,
 *  wait instruction, and atomics its silicon generation guarantees. */
USEARCH_EXPORT typedef enum usearch_cpu_class_t {
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

    /** Diamond Rapids-class x86: the AVX-512 lineage plus AVX10.2's FP8 conversions,
     *  TPAUSE waits, CMPCCXADD conditional adds. */
    usearch_cpu_class_intel_diamond_k,

    /** Graviton2-class Arm - Neoverse N1, the bare NEON floor; phones predating
     *  Armv8.6 land here too. */
    usearch_cpu_class_aws_g2_k,

    /** Armv8.6+ NEON: dot products, bf16 & f16 fused multiplies - Apple M2,
     *  Cortex-A710 & X2 class silicon. */
    usearch_cpu_class_apple_m2_k,

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
     *  addressing that alone lets one module's memory pass 4 GiB. */
    usearch_cpu_class_wasm64_extended_k,
} usearch_cpu_class_t;

/** The GPU families USearch ships tuned variants for, the device twin of
 *  @c usearch_cpu_class_t: one class per generation of kernel kits. */
USEARCH_EXPORT typedef enum usearch_gpu_class_t {
    usearch_gpu_class_unknown_k = 0,

    /** Ampere-class NVIDIA and every GPU without a class of its own - A100, L4, L40S, RTX 30 & 40. */
    usearch_gpu_class_nvidia_ampere_k,

    /** Hopper-class NVIDIA - H100 & H200: no TMA gather, DPX instructions. */
    usearch_gpu_class_nvidia_hopper_k,

    /** Blackwell-class NVIDIA - B200, B300 & GB300: TMA gather4 scattered fetches. */
    usearch_gpu_class_nvidia_blackwell_k,

    /** Consumer Blackwell NVIDIA - RTX 50, RTX PRO 6000 & DGX Spark. */
    usearch_gpu_class_nvidia_blackwellrtx_k,

    /** Rubin-class NVIDIA - the Vera-Rubin platform's device half. */
    usearch_gpu_class_nvidia_rubin_k,
} usearch_gpu_class_t;

/** Where an index's payload lives and which engine traverses it. Selects the active
 *  member of each modality's options union, and decides what "load" means: materialize
 *  into RAM, upload to a device, or open in place on disk. */
USEARCH_EXPORT typedef enum usearch_residency_t {

    /** Volatile-memory index traversed by CPU threads - the concurrent baseline. */
    usearch_residency_ram_k = 0,

    /** Accelerator-memory index traversed by GPU warps on a chosen device. */
    usearch_residency_vram_k = 1,

    /** Persisted index traversed through pluggable I/O engines, sized past RAM. */
    usearch_residency_nvme_k = 2,
} usearch_residency_t;

/**
 *  @brief The host memory domains a RAM-resident index places its shards in - one shard per
 *      named NUMA node, so a dual-socket box names two and gets two graphs whose traversals
 *      stay local to the cores that walk them.
 *
 *  Names where the index lives, never who walks it: worker threads belong to the pool the
 *  bulk verbs take. Every modality's RAM arm is this struct.
 *
 *  Arenas always claim huge pages, falling back to base pages where no pool is provisioned.
 */
USEARCH_EXPORT typedef struct usearch_ram_options_t {

    /** NUMA node ids to place shards on, borrowed for the duration of the call;
     *  @c NULL derives every node the process may allocate on. */
    int32_t const* memory_domains;

    /** Number of entries in @c memory_domains; zero derives. */
    size_t memory_domains_count;
} usearch_ram_options_t;

/**
 *  @brief The devices a VRAM-resident index places its shards in - one shard per named
 *      ordinal, sharded and never replicated, so eight devices host eight slices of one
 *      graph rather than eight copies of it.
 *
 *  Names where the index lives, never who walks it: streams belong to the pool the bulk
 *  verbs take.
 *
 *  Planes are device-resident, never migratable: an index outgrowing the named devices
 *  fails rather than paging to the host, and the answer is to name more devices.
 */
USEARCH_EXPORT typedef struct usearch_vram_options_t {

    /** CUDA device ordinals to place shards on, borrowed for the duration of the
     *  call; @c NULL derives one free device. */
    int32_t const* devices;

    /** Number of entries in @c devices; zero derives. */
    size_t devices_count;
} usearch_vram_options_t;

/**
 *  @brief The devices an NVMe-resident index places its shards on - one shard per named
 *      path, so striping across four drives names four.
 *
 *  Names where the index lives, never who walks it: submission depth belongs to the pool the
 *  bulk verbs take. Naming nothing opens every shard where its manifest already points and
 *  moves no bytes; naming devices makes @c load place shards across them as it reads.
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
USEARCH_EXPORT typedef struct usearch_nvme_options_t {

    /** Devices or directories to place shards on, borrowed for the duration of the
     *  call; @c NULL leaves every shard where its manifest points. */
    usearch_bytes_view_t const* paths;

    /** Number of entries in @c paths; zero derives. */
    size_t paths_count;
} usearch_nvme_options_t;

/** Whether a filtered search admits one candidate key or passes over it. */
USEARCH_EXPORT typedef enum usearch_admission_t {

    /** Skip the key; the walk still routes through it. */
    usearch_reject_k = 0,

    /** Report the key among the matches. */
    usearch_admit_k = 1,
} usearch_admission_t;

/** Type-punned per-key admission predicate for filtered searches. The batched
 *  span-admission form exists only in the C++ layer - a C callback per candidate quad
 *  would cost more than it amortizes. */
USEARCH_EXPORT typedef usearch_admission_t (*usearch_predicate_t)(usearch_key_t, void*);

/** Whether the operation a handler is watching carries on or unwinds. */
USEARCH_EXPORT typedef enum usearch_continuation_t {

    /** Carry on to the next unit of work. */
    usearch_continue_k = 0,

    /** Stop and report @c usearch_error_cancelled_k; whether finished increments stand
     *  is each verb's own contract. */
    usearch_cancel_k = 1,
} usearch_continuation_t;

/**
 *  @brief Periodic report of a long operation's advance, and the caller's chance to stop it.
 *      Counts are in whatever units the verb consumes - rows for an insertion, queries for a
 *      search, shards for a checkpoint - so one handler serves every one of them.
 *
 *  Never invoked concurrently, however many threads perform the work, so the handler needs no
 *  synchronization of its own - and never after the verb it was passed to has returned.
 *
 *  Arguments, in order:
 *  - @c completed: units finished so far, never decreasing across one call's reports.
 *  - @c total: units the call will finish if it runs to completion.
 *  - @c state: the opaque pointer registered alongside the handler, forwarded verbatim.
 */
USEARCH_EXPORT typedef usearch_continuation_t (*usearch_progress_callback_t)( //
    size_t completed, size_t total, void* state);

/**
 *  @brief Everything a call needs to run but an index does not own: the host threads or device
 *      streams that advance the work, and the traversal scratch each of them walks with.
 *
 *  One executor serves any number of index handles, so separate categories of indexes share one
 *  machine's cores or devices instead of each claiming a private set. Its kind is fixed by the
 *  constructor that produced it and must match the residency of every handle it drives; a mismatch
 *  reports @c usearch_error_device_k on the first call.
 *
 *  Scratch is a byte arena carved one slice per thread or stream, sized from geometry rather than from the
 *  scalar type, which is what lets a single executor drive indexes of different quantizations and
 *  metrics. It grows to the widest geometry asked of it and is kept between calls, so a steady
 *  workload allocates once.
 *
 *  An executor admits @b one driver at a time: a second concurrent call reports
 *  @c usearch_error_contended_k rather than corrupting the dispatch already in flight. Handles
 *  stay freely concurrent - it is the executor that serializes, never the index. A single-thread
 *  executor spawns no operating-system threads, so "just use my thread" is one executor per caller
 *  thread rather than a special case in the verbs.
 */
USEARCH_EXPORT typedef void* usearch_executor_t;

/**
 *  @brief Spawns host workers for RAM-resident indexes, interleaved across the compute domains
 *      local to the named memory domains.
 *
 *  @param[in] ram The same options the indexes were initialized with; copied, not borrowed.
 *  @param[in] threads Worker threads; zero takes 7/8 of the cores those domains carry.
 *  @param[out] error Status and a static message; @c usearch_success_k on success.
 *  @return A handle to the spawned executor, or @c NULL on failure.
 *
 *  Passing the very options the indexes were built with is what makes compute and memory domains
 *  agree, leaving no second place to name hardware and no way for the two to disagree.
 *
 *  Interleaving is why a half-size executor takes half of every socket rather than all of one, and
 *  the default stops at 7/8 of the cores because full occupancy starves the ordered publication
 *  path.
 */
USEARCH_EXPORT usearch_executor_t usearch_executor_for_ram( //
    usearch_ram_options_t const* ram, size_t threads, usearch_error_t* error);

/**
 *  @brief Spawns host workers for NVMe-resident indexes, one submission ring apiece, interleaved
 *      across the compute domains local to the devices the paths name.
 *
 *  @param[in] nvme The same options the indexes were initialized with; copied, not borrowed.
 *  @param[in] threads Worker threads; zero derives from the devices' queue capabilities.
 *  @param[out] error Status and a static message; @c usearch_success_k on success.
 *  @return A handle to the spawned executor, or @c NULL on failure.
 *
 *  One ring per thread is what keeps submission lock-free: a thread owns its queue outright, so
 *  an external-memory walk never coordinates with a sibling to issue a read.
 */
USEARCH_EXPORT usearch_executor_t usearch_executor_for_nvme( //
    usearch_nvme_options_t const* nvme, size_t threads, usearch_error_t* error);

/**
 *  @brief Opens one submission stream on every named device, all of them driven by the one host
 *      thread that calls the verbs.
 *
 *  @param[in] vram The same options the indexes were initialized with; copied, not borrowed.
 *  @param[out] error Status and a static message; @c usearch_success_k on success.
 *  @return A handle to the opened executor, or @c NULL on failure.
 *
 *  One stream per device is the whole design: a single host thread keeps every device fed, and
 *  adding host threads would buy nothing, since the work has already left for the accelerator
 *  by the time one of them would return.
 */
USEARCH_EXPORT usearch_executor_t usearch_executor_for_vram( //
    usearch_vram_options_t const* vram, usearch_error_t* error);

/** What an executor does with its threads or streams once a call's work is done. */
USEARCH_EXPORT typedef enum usearch_mood_t {

    /** Park them; the next call pays a wake-up. */
    usearch_mood_chill_k = 0,

    /** Keep them spinning, trading power for the next call's latency. */
    usearch_mood_grind_k = 1,
} usearch_mood_t;

/** Whether a verb returns when its work is done, or when it is merely queued. */
USEARCH_EXPORT typedef enum usearch_completion_t {

    /** Return once the work is finished and its outputs are readable. */
    usearch_completion_blocking_k = 0,

    /** Return once the work is queued; outputs stay indeterminate until a drain. */
    usearch_completion_deferred_k = 1,
} usearch_completion_t;

/**
 *  @brief Optional per-call scheduling of the executor, orthogonal to what a verb computes.
 *      Passing @c NULL blocks and parks, which is what almost every caller wants.
 *
 *  Deferring is not merely faster: it moves both the outputs and the failure out of the call.
 *  Nothing a deferred verb writes may be read, and no error it would have reported is known,
 *  until @c usearch_executor_drain returns.
 */
USEARCH_EXPORT typedef struct usearch_execution_options_t {

    /** What the threads or streams do between calls. */
    usearch_mood_t mood;

    /** Whether the verb waits for its own work. */
    usearch_completion_t completion;
} usearch_execution_options_t;

/**
 *  @brief Waits for every deferred call on @p executor to finish.
 *  @param[inout] executor The handle whose queued work is awaited.
 *  @param[out] error The first failure a drained call reported; @c usearch_success_k if none.
 */
USEARCH_EXPORT void usearch_executor_drain(usearch_executor_t executor, usearch_error_t* error);

/**
 *  @brief Drains every thread or stream, releases the scratch arena, and frees the executor; cannot fail,
 *      so takes no error.
 *  @param[inout] executor The handle to be freed; invalid afterwards.
 */
USEARCH_EXPORT void usearch_executor_free(usearch_executor_t executor);

/**
 *  @brief The canonical lowercase, dash-separated spelling of each closed set's enumerators,
 *      and the inverse that reads one back.
 *
 *  Every binding that touches user text otherwise writes these tables itself, against a header
 *  that nothing checks it against. The names are static storage and are never deallocated; an
 *  unrecognized name reads back as the set's own unknown enumerator.
 */
USEARCH_EXPORT char const* usearch_status_name(usearch_status_t status);

/** Reads a status back from its canonical spelling. */
USEARCH_EXPORT usearch_status_t usearch_status_from_name(usearch_bytes_view_t name);

/** The canonical spelling of a scalar kind, as @c usearch_scalar_f32_k spells "f32". */
USEARCH_EXPORT char const* usearch_scalar_name(usearch_scalar_t scalar_kind);

/** Reads a scalar kind back; @c usearch_scalar_unknown_k when unrecognized. */
USEARCH_EXPORT usearch_scalar_t usearch_scalar_from_name(usearch_bytes_view_t name);

/** The canonical spelling of a residency, as @c usearch_residency_ram_k spells "ram". */
USEARCH_EXPORT char const* usearch_residency_name(usearch_residency_t residency);

/** Reads a residency back; @c usearch_residency_ram_k when unrecognized. */
USEARCH_EXPORT usearch_residency_t usearch_residency_from_name(usearch_bytes_view_t name);

/** The canonical spelling of a CPU class, as the dispatch ladder names its rungs. */
USEARCH_EXPORT char const* usearch_cpu_class_name(usearch_cpu_class_t cpu_class);

/** The canonical spelling of a GPU class, as the dispatch ladder names its rungs. */
USEARCH_EXPORT char const* usearch_gpu_class_name(usearch_gpu_class_t gpu_class);

/**
 *  @brief Retrieves the version of the library.
 *  @return The version of the library as a static null-terminated string.
 */
USEARCH_EXPORT char const* usearch_version(void);

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
