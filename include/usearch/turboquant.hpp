/**
 *  @file       turboquant.hpp
 *  @brief      Umbrella header for TurboQuant integration in USearch.
 *
 *  TurboQuant is a data-oblivious vector quantization algorithm that compresses
 *  dense high-dimensional vectors to 2–4 bits per coordinate with near-optimal
 *  distortion.  It requires no training and supports online insertions.
 *
 *  References:
 *      Zandieh, Daliri, Hadian, Mirrokni.
 *      "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate"
 *      arXiv:2504.19874, ICLR 2026.
 *
 *  Architecture:
 *      turboquant_rotation.hpp  — HD³ (randomized Hadamard) rotation
 *      turboquant_codec.hpp     — Lloyd-Max codebooks + encode/decode
 *      turboquant_metric.hpp    — Distance functions on compressed vectors
 */
#pragma once

#include <usearch/turboquant_rotation.hpp>
#include <usearch/turboquant_codec.hpp>
#include <usearch/turboquant_metric.hpp>
