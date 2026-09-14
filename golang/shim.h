/**
 *  @file golang/shim.h
 *  @author Ash Vardanian
 *  @date August 16, 2026
 *  @brief Declarations the Go binding needs and the C header cannot give it: accessors for the
 *      anonymous residency union inside @c usearch_dense_options_t, and the Go-side admission
 *      trampoline the filtered search installs as a C callback.
 *
 *  cgo renders an anonymous union as unnamed storage no Go expression can address, so the
 *  union is crossed through these functions rather than reconstructed field by field. Only
 *  the union is opaque - each arm is a named struct cgo handles like any other, so the arms
 *  travel whole and nothing here takes the union apart. The definitions live in @c shim.c
 *  because a Go file carrying an @c //export directive may hold only declarations in its
 *  preamble.
 */
#ifndef UNUM_USEARCH_GO_SHIM_H
#define UNUM_USEARCH_GO_SHIM_H

#include "usearch/dense_index.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Tags @p options as RAM-resident and fills that arm of the union. */
void usearch_go_write_ram(usearch_dense_options_t* options, usearch_ram_options_t ram);

/** Tags @p options as VRAM-resident and fills that arm of the union. */
void usearch_go_write_vram(usearch_dense_options_t* options, usearch_vram_options_t vram);

/** Tags @p options as NVMe-resident and fills that arm of the union. */
void usearch_go_write_nvme(usearch_dense_options_t* options, usearch_nvme_options_t nvme);

/** Reads the RAM arm; the caller checks @c options->residency first. */
usearch_ram_options_t usearch_go_read_ram(usearch_dense_options_t const* options);

/** Reads the VRAM arm; the caller checks @c options->residency first. */
usearch_vram_options_t usearch_go_read_vram(usearch_dense_options_t const* options);

/** Reads the NVMe arm; the caller checks @c options->residency first. */
usearch_nvme_options_t usearch_go_read_nvme(usearch_dense_options_t const* options);

/**
 *  @brief The admission callback Go defines and C calls - the one symbol here that crosses in
 *      that direction, which is why it alone wears Go's spelling.
 *
 *  cgo emits a C symbol named exactly as the Go function, and Go spells an unexported function
 *  in camel case, so the name is not this file's to choose. Everything else here is C that Go
 *  calls, and reads as C.
 */
usearch_admission_t usearchDenseAdmits(usearch_key_t key, void* state);

/** Hands back @c usearchDenseAdmits as a function pointer. cgo renders a C function as
 *  a Go wrapper rather than an address, so the pointer has to be taken on this side. */
usearch_predicate_t usearch_go_admits(void);

#ifdef __cplusplus
}
#endif

#endif // UNUM_USEARCH_GO_SHIM_H
