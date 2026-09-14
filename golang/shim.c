/**
 *  @file golang/shim.c
 *  @author Ash Vardanian
 *  @date August 16, 2026
 *  @brief Accessors for the anonymous residency union of @c usearch_dense_options_t.
 *
 *  Each writer stamps the residency tag alongside its arm, so a caller cannot leave the tag
 *  disagreeing with the member it filled.
 */
#include "shim.h"

void usearch_go_write_ram(usearch_dense_options_t* options, usearch_ram_options_t ram) {
    options->residency = usearch_residency_ram_k;
    options->ram = ram;
}

void usearch_go_write_vram(usearch_dense_options_t* options, usearch_vram_options_t vram) {
    options->residency = usearch_residency_vram_k;
    options->vram = vram;
}

void usearch_go_write_nvme(usearch_dense_options_t* options, usearch_nvme_options_t nvme) {
    options->residency = usearch_residency_nvme_k;
    options->nvme = nvme;
}

usearch_ram_options_t usearch_go_read_ram(usearch_dense_options_t const* options) { return options->ram; }

usearch_vram_options_t usearch_go_read_vram(usearch_dense_options_t const* options) { return options->vram; }

usearch_nvme_options_t usearch_go_read_nvme(usearch_dense_options_t const* options) { return options->nvme; }

usearch_predicate_t usearch_go_admits(void) { return &usearchDenseAdmits; }
