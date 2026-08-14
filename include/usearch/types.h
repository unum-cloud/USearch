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

/**
 *  @brief User-facing 64-bit key identifying one entry of any index. Keys need not be
 *      unique: several entries may share one, and count-like functions report how many.
 */
USEARCH_EXPORT typedef uint64_t usearch_key_t;

/**
 *  @brief Distance or similarity score between two entries, in the metric of the index
 *      that produced it. Wide enough for every modality's scoring range.
 */
USEARCH_EXPORT typedef double usearch_distance_t;

/**
 *  @brief Pointer to a null-terminated error message, or @c NULL when the call succeeded.
 *      Returned error messages are static and @b don't need to be deallocated.
 */
USEARCH_EXPORT typedef char const* usearch_error_t;

/**
 *  @brief Scalar types the indexes store and exchange, named by width and encoding.
 *      Sub-byte types pack several dimensions per byte, as their @c xN suffixes spell.
 */
USEARCH_EXPORT typedef enum usearch_scalar_t {
    usearch_scalar_unknown_k = 0,

    usearch_scalar_f64_k = 1,
    usearch_scalar_f32_k = 2,
    usearch_scalar_bf16_k = 3,
    usearch_scalar_f16_k = 4,
    usearch_scalar_e5m2_k = 5,
    usearch_scalar_e4m3_k = 6,
    usearch_scalar_e3m2_k = 7,
    usearch_scalar_e2m3_k = 8,

    usearch_scalar_i8_k = 10,
    usearch_scalar_i4x2_k = 11,

    usearch_scalar_u8_k = 20,
    usearch_scalar_u4x2_k = 21,
    usearch_scalar_u1x8_k = 22,

} usearch_scalar_t;

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
