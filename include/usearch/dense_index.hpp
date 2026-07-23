/**
 *  @brief The dense modality's flat entry point - equidimensional vectors under
 *      inner-product-derived metrics, across every residency.
 *  @author Ash Vardanian
 *  @file include/usearch/dense_index.hpp
 *  @date July 23, 2026
 *
 *  Mirrors the sibling libraries' glue idiom - one flat header beside a same-named
 *  directory - so a reader opens @c dense/ to see everything about the modality, with
 *  residency and capability visible as the file names inside it. Including this header
 *  pulls the RAM residency: the CPU baseline every platform owns. The VRAM and NVMe
 *  siblings stay opt-in through their own extensions - @c dense/vram_index.cuh rides the
 *  CUDA SDK and @c dense/nvme_index.hpp the platform I/O rings, so neither belongs in a
 *  portable default include.
 */
#pragma once

#include "usearch/dense/ram_index.hpp"
#include "usearch/dense/ram_indexes.hpp"
#include "usearch/dense/ram_metric.hpp"
