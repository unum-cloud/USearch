/**
 *  @file src/sequence/cpu_variants_serial.cpp
 *  @author Ash Vardanian
 *  @date October 13, 2026
 *  @brief The serial unit of the sequence variants, on the host and on storage: the floor every
 *      ladder ends on, at the baseline flags.
 */
#include "sequence_specializations.cuh"

template struct basic_class_tables<sequence_traits_t, dense_home_t::ram_k, cpu_class_t::serial_k>;
template struct basic_class_tables<sequence_traits_t, dense_home_t::nvme_k, cpu_class_t::serial_k>;
