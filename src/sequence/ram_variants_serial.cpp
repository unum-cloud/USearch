/**
 *  @file src/sequence/ram_variants_serial.cpp
 *  @author Ash Vardanian
 *  @date October 13, 2026
 *  @brief The serial unit of the sequence arms: the floor every ladder ends on, at the baseline
 *      flags.
 */
#include "sequence_specializations.cuh"

template struct sequence_ram_variant<cpu_class_t::serial_k>;
template struct sequence_nvme_variant<cpu_class_t::serial_k>;
