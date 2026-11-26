/**
 *  @file include/usearch/sequence/alphabet.hpp
 *  @author Ash Vardanian
 *  @date October 22, 2026
 *  @brief The byte alphabet a sequence is spelled over, and the similarities a protein alignment
 *      and a DNA overlap score with unless an index names its own.
 */
#pragma once

#include <cstddef> // `std::size_t`
#include <cstdint> // `std::int8_t`

#include <span>        // `std::span`
#include <string_view> // `std::string_view`

namespace unum::usearch::sequence {

/** Bytes a code of a byte alphabet spans, the side of the square matrix a fold reads. */
inline constexpr std::size_t alignment_byte_alphabet_k = std::size_t {1} << 8;

/** A row-major square of similarities between bytes, as a fold reads one. */
using byte_similarities_span_t = std::span<std::int8_t, alignment_byte_alphabet_k * alignment_byte_alphabet_k>;

/** The proteins' classes in BLOSUM62's order: 20 amino acids, then B, Z, X and the stop. */
inline constexpr std::string_view protein_letters_k = "ARNDCQEGHILKMFPSTWYVBZX*";

/** BLOSUM62 over @ref protein_letters_k, as NCBI publishes it. */
inline constexpr std::int8_t blosum62_k[24][24] = {
    {4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0, -2, -1, 0, -4},
    {-1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3, -1, 0, -1, -4},
    {-2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3, 3, 0, -1, -4},
    {-2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3, 4, 1, -1, -4},
    {0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, -3, -3, -2, -4},
    {-1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2, -1, -2, 0, 3, -1, -4},
    {-1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -2, 1, 4, -1, -4},
    {0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3, -1, -2, -1, -4},
    {-2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3, 0, 0, -1, -4},
    {-1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -2, -1, -3, -1, 3, -3, -3, -1, -4},
    {-1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1, -4, -3, -1, -4},
    {-1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3, -2, -2, 0, 1, -1, -4},
    {-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1, -3, -1, -1, -4},
    {-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1, -3, -3, -1, -4},
    {-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -3, -2, -2, -1, -2, -4},
    {1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2, 0, 0, 0, -4},
    {0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0, -1, -1, 0, -4},
    {-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11, 2, -3, -4, -3, -2, -4},
    {-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7, -1, -3, -2, -1, -4},
    {0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4, -3, -2, -1, -4},
    {-2, -1, 3, 4, -3, 0, 1, -1, 0, -3, -4, 0, -3, -3, -2, 0, -1, -4, -3, -3, 4, 1, -1, -4},
    {-1, 0, 0, 1, -3, 3, 4, -2, 0, -3, -3, 1, -1, -3, -1, 0, -1, -3, -2, -2, 1, 4, -1, -4},
    {0, -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2, 0, 0, -2, -1, -1, -1, -1, -1, -4},
    {-4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, 1},
};

/** The gap cost a protein alignment charges beside BLOSUM62. */
inline constexpr std::int8_t protein_gap_k = 7;

/** The class of @p byte among @ref protein_letters_k: a letter's own in either case, and that of
 *  @c X for any other byte. */
constexpr std::size_t protein_class(std::size_t byte) noexcept {
    char const letter = static_cast<char>(byte >= 'a' && byte <= 'z' ? byte - ('a' - 'A') : byte);
    std::size_t const found = protein_letters_k.find(letter);
    return found != std::string_view::npos ? found : protein_letters_k.find('X');
}

/** Writes BLOSUM62 spread over bytes into @p similarities, each byte scoring as its class does. */
constexpr void protein_similarities(byte_similarities_span_t similarities) noexcept {
    for (std::size_t row = 0; row != alignment_byte_alphabet_k; ++row)
        for (std::size_t column = 0; column != alignment_byte_alphabet_k; ++column)
            similarities[row * alignment_byte_alphabet_k + column] =
                blosum62_k[protein_class(row)][protein_class(column)];
}

/** The nucleotides' classes: purines take even places and pyrimidines odd ones. */
inline constexpr std::string_view dna_letters_k = "ACGT";

/** The gap cost a DNA overlap charges beside its own similarities. */
inline constexpr std::int8_t dna_gap_k = 2;

/** The class of @p byte among @ref dna_letters_k: a letter's own in either case, @c U read as
 *  @c T, and that of @c T for any other byte. */
constexpr std::size_t dna_class(std::size_t byte) noexcept {
    char const letter = static_cast<char>(byte >= 'a' && byte <= 'z' ? byte - ('a' - 'A') : byte);
    std::size_t const found = dna_letters_k.find(letter);
    return found != std::string_view::npos ? found : dna_letters_k.find('T');
}

/** Writes the nucleotides' similarities spread over bytes into @p similarities: two for a match,
 *  one for a transition, which keeps a purine a purine, and none for a transversion. */
constexpr void dna_similarities(byte_similarities_span_t similarities) noexcept {
    for (std::size_t row = 0; row != alignment_byte_alphabet_k; ++row)
        for (std::size_t column = 0; column != alignment_byte_alphabet_k; ++column) {
            std::size_t const first = dna_class(row), second = dna_class(column);
            similarities[row * alignment_byte_alphabet_k + column] = first == second               ? 2
                                                                     : (first & 1) == (second & 1) ? 1
                                                                                                   : 0;
        }
}

/** The letter pairing with the nucleotide @p letter on the other strand, in its case, @c U read as
 *  @c T; zero for any other byte. */
constexpr char dna_complement(char letter) noexcept {
    switch (letter) {
    case 'A': return 'T';
    case 'C': return 'G';
    case 'G': return 'C';
    case 'T':
    case 'U': return 'A';
    case 'a': return 't';
    case 'c': return 'g';
    case 'g': return 'c';
    case 't':
    case 'u': return 'a';
    default: return 0;
    }
}

} // namespace unum::usearch::sequence
