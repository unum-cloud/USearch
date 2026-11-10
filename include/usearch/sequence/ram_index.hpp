/**
 *  @file include/usearch/sequence/ram_index.hpp
 *  @author Ash Vardanian
 *  @date August 28, 2026
 *  @brief Similarity Search over variable-length byte sequences - text, DNA, or protein.
 *
 *  A stored payload is an @c sz::string its node record owns, built from the index's own
 *  allocator, so a string of up to 22 bytes lives inline and a longer one costs one block.
 *  Scoring is batched: one @c reset → @c push → @c drain round becomes a single sweep over
 *  the whole graph fan-out, through the metrics of @c ram_metric.hpp.
 *
 *  @section sequence_ram_index_concurrency Concurrency & Transactions
 *
 *  A value is moved into its record once, at publication, and only read after, so a view a
 *  metric stages stays valid for the round. Each metric instance belongs to one thread;
 *  the index spreads rounds across threads.
 *
 *  @section sequence_ram_index_persistence Persistence
 *
 *  Checkpoints are SafeGraphs monoliths: keys, @c utf8 values, and neighbor lists, with the
 *  kernel the graph was built under recorded verbatim. They are written and read through
 *  @c safegraphs_write_index and @c safegraphs_read_index with the
 *  @ref unum::usearch::sequence::codec_t, whose @c load assigns every value into its record's own
 *  string.
 *
 *  @section sequence_ram_index_organization File Organization
 *
 *  @b Serialization - the status mapping and the codec. @b Index - the alias over the basic
 *  graph index.
 */
#pragma once

#include <cstddef> // `std::size_t`

#include <memory>      // `std::allocator`, `std::allocator_traits`
#include <span>        // `std::span`, `std::as_bytes`
#include <string_view> // `std::string_view` - a stored sequence, borrowed

#include <forkunion/atomics.hpp> // `fu::standard_atomic_ref` - the default reference

#include <stringzilla/stringzilla.hpp> // `sz::string`, `sz::status_t` - the stored values

#include "usearch/utilities/types.hpp"
#include "usearch/utilities/safegraphs_format.hpp" // SafeGraphs monoliths - the checkpoints
#include "usearch/basic/ram_index.hpp"
#include "usearch/sequence/ram_metric.hpp"

namespace unum::usearch::sequence {

#pragma region Serialization

/**
 *  @brief USearch's reading of a StringZilla status, the one crossing between the two vocabularies.
 *
 *  @par Statuses:
 *  - @c overflow_k StringZilla risked an overflow.
 *  - @c shape_mismatch_k StringZilla found unexpected dimensions.
 */
inline status_t status_of(sz::status_t status) noexcept {
    switch (status) {
    case sz::status_t::success_k: return status_t::success_k;
    case sz::status_t::bad_alloc_k: return status_t::bad_alloc_k;
    case sz::status_t::overflow_risk_k: return status_t::overflow_k;
    case sz::status_t::unexpected_dimensions_k: return status_t::shape_mismatch_k;
    default: return status_t::internal_k;
    }
}

/** Bytes out as they are, bytes in to the record's own string through its own allocator. */
struct codec_t {

    /** The interchange token of the elements a value emits: raw UTF-8 bytes. */
    inline static constexpr dtype_t value_dtype_k = dtype_t::utf8_k;

    template <typename string_type_>
    std::size_t storage_requirements(string_type_ const& value) const noexcept {
        return value.size();
    }

    /**
     *  Copies the bytes of @p value into @p output as they are.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k @p output differs in length from @p value.
     */
    template <typename string_type_>
    status_t save(string_type_ const& value, bytes_span_t output) const noexcept {
        return copy_bytes(output, std::as_bytes(std::span {value}));
    }

    /** Assigns the bytes of @p input to @p value through the string's own allocator. */
    template <typename string_type_>
    status_t load(bytes_view_t input, string_type_& value) const noexcept {
        return status_of(value.assign(chars_of(input)));
    }

    /** What @ref view hands out for a stored sequence. */
    using view_t = std::string_view;

    /** The string a stored @p sequence holds, borrowed. */
    view_t view(bytes_view_t sequence) const noexcept { return chars_of(sequence); }
};

/** Similarity Search over byte sequences: owned strings, integer distances, and a metric chosen per
 *  call - so a graph built under one kernel is queried under another without rebuilding. */
template <typename key_type_ = uuid128_t, typename node_id_type_ = u32_t, typename wait_type_ = dummy_waiter_t,
          template <typename> class atomic_ref_template_ = fu::standard_atomic_ref,
          typename allocator_type_ = std::allocator<byte_t>, typename cache_hints_type_ = dummy_cache_hints_t>
using ram_index =
    basic::ram_index<sz::string<typename std::allocator_traits<allocator_type_>::template rebind_alloc<char>>,
                     key_type_, node_id_type_, sz_size_t, wait_type_, atomic_ref_template_, allocator_type_,
                     cache_hints_type_>;

using ram_index_t = ram_index<>;

#pragma endregion Serialization

} // namespace unum::usearch::sequence
