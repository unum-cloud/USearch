/**
 *  @file src/dense/ram_variants_serial.cpp
 *  @author Ash Vardanian
 *  @date September 7, 2026
 *  @brief The serial unit: every cell at the portable floor, no instruction set beyond the
 *      baseline.
 */
#include "dense_specializations.cuh"

template struct specialized_index<ram_indexes<nk_bf16_k, cpu_class_t::serial_k>>;
#if USEARCH_DENSE_DTYPES >= 2
template struct specialized_index<ram_indexes<nk_u1_k, cpu_class_t::serial_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 3
template struct specialized_index<ram_indexes<nk_i8_k, cpu_class_t::serial_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 4
template struct specialized_index<ram_indexes<nk_e4m3_k, cpu_class_t::serial_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 5
template struct specialized_index<ram_indexes<nk_f16_k, cpu_class_t::serial_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 6
template struct specialized_index<ram_indexes<nk_u8_k, cpu_class_t::serial_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 7
template struct specialized_index<ram_indexes<nk_e5m2_k, cpu_class_t::serial_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 8
template struct specialized_index<ram_indexes<nk_i4_k, cpu_class_t::serial_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 9
template struct specialized_index<ram_indexes<nk_u4_k, cpu_class_t::serial_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 10
template struct specialized_index<ram_indexes<nk_e3m2_k, cpu_class_t::serial_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 11
template struct specialized_index<ram_indexes<nk_e2m3_k, cpu_class_t::serial_k>>;
#endif
