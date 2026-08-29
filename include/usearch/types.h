/**
 *  @brief Shared vocabulary of the USearch C interface: scalar types, keys, distances,
 *      error reporting, residency selection, and library introspection - everything the
 *      modality headers - @c dense_index.h, @c sequence_index.h, @c geospatial_index.h -
 *      have in common and nothing they don't.
 *
 *  @author Ash Vardanian
 *  @file include/usearch/types.h
 *
 *  Include a modality header directly - it pulls this one in; there is no umbrella header,
 *  matching how NumKong's per-family headers are consumed without @c numkong.h. The C
 *  vocabulary sits beside its C++ twin @c types.hpp, so one installed tree serves both
 *  languages and the headers pair by stem.
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

/** @brief The library's version, which the release tooling rewrites in place. */
#define USEARCH_VERSION_MAJOR 3
#define USEARCH_VERSION_MINOR 0
#define USEARCH_VERSION_PATCH 0

/**
 *  @brief User-facing 128-bit key identifying one entry of any index - sixteen opaque
 *      bytes, rendered canonically as a lowercase dashed UUID, matching SafeGraphs'
 *      @c bin16 key column. Keys need not be unique: several entries may share one, and
 *      count-like functions report how many. Integer identities park in the low bytes
 *      with the rest zeroed.
 */
USEARCH_EXPORT typedef struct usearch_key_t {
    uint8_t bytes[16];
} usearch_key_t;

/**
 *  @brief Distance or similarity score between two entries, in the metric of the index
 *      that produced it. Wide enough for every modality's scoring range.
 */
USEARCH_EXPORT typedef double usearch_distance_t;

/**
 *  @brief A borrowed run of bytes and its length, never null-terminated by contract - the
 *      shape StringZilla's @c sz_string_view_t takes and the one every modality's string
 *      arguments arrive in. Callers holding a Rust @c str, a Go string, or a Java byte array
 *      pass their own storage without manufacturing a terminator none of them keeps.
 */
USEARCH_EXPORT typedef struct usearch_string_t {
    char const* data;
    size_t length;
} usearch_string_t;

/**
 *  @brief What went wrong, in a vocabulary shared by the engine and by every binding that
 *      validates before calling it - so one failure carries one identity, whoever catches it.
 *
 *  Some enumerators name failures this library never returns itself: a binding holding richer
 *  layout information than the ABI can accept rejects the call before it crosses, and reports
 *  the same status the engine would if it could see the same thing. The set describes the
 *  project's failures, not this implementation's exits, so unused enumerators are deliberate.
 */
USEARCH_EXPORT typedef enum usearch_status_t {
    /** @brief The call completed; @c message carries nothing. */
    usearch_success_k = 0,
    /** @brief An allocation or reservation could not be satisfied. */
    usearch_error_out_of_memory_k = 1,
    /** @brief Dimensionality, row counts, or output capacity disagree. */
    usearch_error_shape_mismatch_k = 2,
    /** @brief A stride or packing the engine cannot traverse. */
    usearch_error_unsupported_layout_k = 3,
    /** @brief A scalar, metric, or residency tag this build does not know. */
    usearch_error_unknown_kind_k = 4,
    /** @brief A mutating verb on a handle that opens its checkpoint in place. */
    usearch_error_read_only_k = 5,
    /** @brief A file could not be opened, read, written, or parsed. */
    usearch_error_io_k = 6,
    /** @brief A device could not be selected, or its memory could not be claimed. */
    usearch_error_device_k = 7,
    /** @brief A caller-supplied representation could not be decoded - a malformed key, a
     *      truncated buffer, a checkpoint that ends early. */
    usearch_error_malformed_input_k = 8,
    /** @brief The library reported neither success nor a failure - a contract violation
     *      rather than a condition, so it names a defect and not a state. */
    usearch_error_internal_k = 9,
    /** @brief A hard design ceiling was reached, like the 32-bit id space; a smaller request
     *      cannot help where a smaller allocation would. */
    usearch_error_overflow_k = 10,
    /** @brief The verb needs a prior successful reservation that never happened. */
    usearch_error_uninitialized_k = 11,
    /** @brief The request contradicts an already-frozen configuration, like re-reserving with
     *      a different geometry. */
    usearch_error_config_mismatch_k = 12,
    /** @brief A synchronization contract was violated by a concurrent operation; discard any
     *      produced output and retry in a quieter phase. */
    usearch_error_contended_k = 13,
} usearch_status_t;

/**
 *  @brief Outcome of one call: a status every binding can branch on, and a static message
 *      carrying the detail the status cannot. Messages are static and @b don't need to be
 *      deallocated, and are @c NULL exactly when the status is @c usearch_success_k.
 */
USEARCH_EXPORT typedef struct usearch_error_t {
    usearch_status_t status;
    char const* message;
} usearch_error_t;

/**
 *  @brief Scalar types the indexes store and exchange, named by width and encoding.
 *      Sub-byte types pack several dimensions per byte, as their @c xN suffixes spell.
 */
USEARCH_EXPORT typedef enum usearch_scalar_t {
/** @brief No scalar named - an unset field, or a custom metric owning its own storage. */
    usearch_scalar_unknown_k = 0,

/** @brief IEEE-754 binary64: ±1.80e308, epsilon 2.22e-16, subnormals to 4.94e-324,
     *      with infinities and NaNs. */
    usearch_scalar_f64_k = 1,
/** @brief IEEE-754 binary32: ±3.40e38, epsilon 1.19e-7, subnormals to 1.40e-45, with
     *      infinities and NaNs. */
    usearch_scalar_f32_k = 2,
/** @brief Truncated f32 - 8-bit exponent, 7-bit mantissa: f32's range at ±3.39e38 and
     *      epsilon 7.81e-3, with infinities and NaNs. */
    usearch_scalar_bf16_k = 3,
/** @brief IEEE-754 binary16: ±65504, epsilon 9.77e-4, subnormals to 5.96e-8, with
     *      infinities and NaNs. */
    usearch_scalar_f16_k = 4,
/** @brief FP8 with a 5-bit exponent: ±57344, epsilon 0.25, with infinities and NaNs -
     *      f16's dynamic range carried on two mantissa bits. */
    usearch_scalar_e5m2_k = 5,
/** @brief FP8 with a 4-bit exponent: ±448, epsilon 0.125, no infinities, and two NaN
     *      codes at 0x7F and 0xFF. */
    usearch_scalar_e4m3_k = 6,
/** @brief FP6 with a 3-bit exponent, byte-aligned with two padding bits: ±28, no
     *      infinities or NaNs, so every one of its 64 codes is a number. */
    usearch_scalar_e3m2_k = 7,
/** @brief FP6 with a 2-bit exponent, byte-aligned with two padding bits: ±7.5, no
     *      infinities or NaNs, and 72% of its codes outside [-1, 1]. */
    usearch_scalar_e2m3_k = 8,

/** @brief 8-bit signed integer spanning -128 to 127, every code a number. */
    usearch_scalar_i8_k = 10,
/** @brief Two signed nibbles per byte spanning -8 to 7, even dimensions in the low
     *      nibble and odd ones in the high. */
    usearch_scalar_i4x2_k = 11,

/** @brief 8-bit unsigned integer spanning 0 to 255, every code a number. */
    usearch_scalar_u8_k = 20,
/** @brief Two unsigned nibbles per byte spanning 0 to 15, even dimensions in the low
     *      nibble and odd ones in the high. */
    usearch_scalar_u4x2_k = 21,
/** @brief Eight single-bit dimensions per byte - the form the Hamming & Jaccard metrics
     *      score, where only the set-bit count of a XOR or AND carries meaning. */
    usearch_scalar_u1x8_k = 22,

} usearch_scalar_t;

/**
 *  @brief The CPU families USearch ships tuned variants for - dispatch reports its pick in
 *      this vocabulary, and callers may pin one. Each class bundles the compute features,
 *      wait instruction, and atomics its silicon generation guarantees.
 */
USEARCH_EXPORT typedef enum usearch_cpu_class_t {
    usearch_cpu_class_unknown_k = 0,
    /** @brief The portable floor: no SIMD assumptions, standard yields. */
    usearch_cpu_class_serial_k,
    /** @brief Haswell-class x86: AVX2 & FMA, the 256-bit floor. */
    usearch_cpu_class_intel_haswell_k,
    /** @brief Skylake-class x86: the AVX-512 foundation tier. */
    usearch_cpu_class_intel_skylake_k,
    /** @brief Icelake-class x86: AVX-512 VNNI & VPOPCNT - the integer & bit workhorse. */
    usearch_cpu_class_intel_icelake_k,
    /** @brief Genoa-class x86 - Zen 4 and newer: native bf16 dot products over AVX-512. */
    usearch_cpu_class_amd_genoa_k,
    /** @brief Graviton2-class Arm - Neoverse N1, the bare NEON floor; phones predating
     *      Armv8.6 land here too. */
    usearch_cpu_class_aws_g2_k,
    /** @brief Armv8.6+ NEON: dot products, bf16 & f16 fused multiplies - Apple M2,
     *      Cortex-A710 & X2 class silicon. */
    usearch_cpu_class_apple_m2_k,
    /** @brief Graviton5-class Arm - Neoverse V3, Armv9.2: monitored waits guaranteed. */
    usearch_cpu_class_aws_g5_k,
    /** @brief Vera-class Arm - NVIDIA's Armv9.2-generation cores with FP8 dot products. */
    usearch_cpu_class_nvidia_vera_k,
} usearch_cpu_class_t;

/**
 *  @brief The GPU families USearch ships tuned variants for, the device twin of
 *      @c usearch_cpu_class_t: one class per generation of kernel kits.
 */
USEARCH_EXPORT typedef enum usearch_gpu_class_t {
    usearch_gpu_class_unknown_k = 0,
    /** @brief Hopper-class NVIDIA - H100 & H200: no TMA gather, DPX instructions. */
    usearch_gpu_class_nvidia_hopper_k,
    /** @brief Blackwell-class NVIDIA - B200 & GB200: TMA gather4 scattered fetches. */
    usearch_gpu_class_nvidia_blackwell_k,
    /** @brief Rubin-class NVIDIA - the Vera-Rubin platform's device half. */
    usearch_gpu_class_nvidia_rubin_k,
} usearch_gpu_class_t;

/**
 *  @brief Where an index's payload lives and which engine traverses it. Selects the active
 *      member of each modality's options union, and decides what "load" means: materialize
 *      into RAM, upload to a device, or open in place on disk.
 */
USEARCH_EXPORT typedef enum usearch_residency_t {
    /** @brief Volatile-memory index traversed by CPU threads - the concurrent baseline. */
    usearch_residency_ram_k = 0,
    /** @brief Accelerator-memory index traversed by GPU warps on a chosen device. */
    usearch_residency_vram_k = 1,
    /** @brief Persisted index traversed through pluggable I/O engines, sized past RAM. */
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
    /** @brief NUMA node ids to place shards on, borrowed for the duration of the call;
     *      @c NULL derives every node the process may allocate on. */
    int32_t const* memory_domains;
    /** @brief Number of entries in @c memory_domains; zero derives. */
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
    /** @brief CUDA device ordinals to place shards on, borrowed for the duration of the
     *      call; @c NULL derives one free device. */
    int32_t const* devices;
    /** @brief Number of entries in @c devices; zero derives. */
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
    /** @brief Devices or directories to place shards on, borrowed for the duration of the
     *      call; @c NULL leaves every shard where its manifest points. */
    usearch_string_t const* paths;
    /** @brief Number of entries in @c paths; zero derives. */
    size_t paths_count;
} usearch_nvme_options_t;

/**
 *  @brief Type-punned per-key admission predicate for filtered searches: return non-zero
 *      to admit the key. The batched span-admission form exists only in the C++ layer -
 *      a C callback per candidate quad would cost more than it amortizes.
 */
USEARCH_EXPORT typedef int (*usearch_predicate_t)(usearch_key_t, void*);

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
