/**
 *  @file include/usearch/sequence/ram_index.hpp
 *  @author Ash Vardanian
 *  @date August 28, 2026
 *  @brief Similarity Search over variable-length byte sequences - text, DNA, or protein.
 *
 *  A stored payload is an @c sz::string its node record owns, built from the index's own
 *  allocator: a sketch of the bytes the index names, then the sequence, in one block. Inserts and
 *  refinements score the sketches, searches walk the sketches first and then score the sequences
 *  canonically from the nodes that walk found, and all of them read one value.
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
 *  Checkpoints are SafeGraphs monoliths: keys, @c utf8 values, and neighbor lists, with the kernel
 *  the graph was built under recorded verbatim. They hold the sequences alone, written and read
 *  through @c safegraphs_write_index and @c safegraphs_read_index with the @c stored_codec, whose
 *  @c load writes every sequence into the string of its record, behind a sketch derived in place
 *  where the kit sketches.
 *  A view reads values no sketch precedes through the plain @c codec_t.
 *
 *  @section sequence_ram_index_organization File Organization
 *
 *  @b Serialization - the status mapping, the codec and the alias over the basic graph index.
 *  @b Stored @b Values - the layout of a sketch and its sequence, the metrics each role scores a
 *  stored value with, and the codecs of an index storing sketches.
 */
#pragma once

#include <cstddef> // `std::size_t`, `std::to_integer` - a tape value's closing count
#include <cstring> // `std::memcpy`, `std::memset` - a stored sequence and its closing padding

#include <limits>      // `std::numeric_limits` - a round past 32-bit slots
#include <memory>      // `std::allocator`, `std::allocator_traits`
#include <span>        // `std::span`, `std::as_bytes`
#include <string_view> // `std::string_view` - a stored sequence, viewed

#include <forkunion/atomics.hpp> // `fu::standard_atomic_ref` - the default reference

#include <stringzilla/string.hpp> // `sz::string`, `sz::status_t` - the stored values

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
    case sz::status_t::success_k: return status_code_t::success_k;
    case sz::status_t::bad_alloc_k: return status_code_t::out_of_memory_k;
    case sz::status_t::overflow_risk_k: return status_code_t::overflow_k;
    case sz::status_t::unexpected_dimensions_k: return status_code_t::shape_mismatch_k;
    default: return status_code_t::internal_k;
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

    /** The string a stored @p sequence holds, viewed. */
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

#pragma region Stored Values

/** Bytes a sequence of @p bytes takes in a device tape behind a sketch of @p sketch_bytes: the
 *  two, then padding to a multiple of four that always holds its own closing count. */
constexpr std::size_t vram_stored_bytes(std::size_t bytes, std::size_t sketch_bytes) noexcept {
    return round_up_to_multiple<std::size_t>(sketch_bytes + bytes + 1, 4);
}

/** The sequence inside the tape value @p stored behind a sketch of @p sketch_bytes. */
inline bytes_view_t vram_stored_sequence(bytes_view_t stored, std::size_t sketch_bytes) noexcept {
    std::size_t const closing = std::to_integer<std::size_t>(stored.back());
    return stored.subspan(sketch_bytes, stored.size() - sketch_bytes - closing);
}

/** Whether every value of @p tape is laid out as a device stores it behind a sketch of
 *  @p sketch_bytes: a multiple of four bytes, the sketch and a closing count of one to four. */
inline bool vram_tape_is_stored(tape_view<byte_t const> tape, std::size_t sketch_bytes) noexcept {
    for (std::size_t entry = 0; entry != tape.size(); ++entry) {
        bytes_view_t const stored = tape[entry];
        if (stored.size() < sketch_bytes + 4 || stored.size() % 4) return false;
        std::size_t const closing = std::to_integer<std::size_t>(stored.back());
        if (closing < 1 || closing > 4) return false;
    }
    return true;
}

/** Closes the tape value @p stored over a sequence of @p bytes: zero padding, then its count. */
inline void vram_close_stored(bytes_span_t stored, std::size_t bytes, std::size_t sketch_bytes) noexcept {
    std::size_t const closing = stored.size() - sketch_bytes - bytes;
    std::memset(stored.data() + sketch_bytes + bytes, 0, closing);
    stored.back() = static_cast<byte_t>(closing);
}

/** Whether the device kit @p kit_ writes sketches, as host code sees it, where the kit's device
 *  verbs take no arguments it could name. */
template <typename kit_>
concept a_device_sketching_kit = requires { kit_::sketch_distance; };

/** The canonical metric over values that keep a sketch of the index's bytes in front of each
 *  sequence, or none for a kit that writes none: a stored @p value_type_ loses the sketch on its
 *  way in, and any other query form passes as it is. */
template <typename metric_type_, typename value_type_>
class stored_canonical_metric {
  public:
    using kit_t = typename metric_type_::kit_t;
    using distance_t = typename metric_type_::distance_t;
    using allocator_t = typename metric_type_::allocator_t;
    using query_view_t = typename metric_type_::query_view_t;
    using is_transparent = void;

    stored_canonical_metric(allocator_t const& allocator = {}) noexcept : canonical_(allocator) {}

    /** Readies the canonical metric with the arguments it takes itself, past sketches of
     *  @p sketch_bytes bytes. */
    template <typename... arguments_types_>
    status_t init(std::size_t staging_limit, std::size_t sketch_bytes, arguments_types_ const&... arguments) noexcept {
        sketch_bytes_ = sketch_bytes;
        return canonical_.init(staging_limit, arguments...);
    }
    status_t status() const noexcept { return canonical_.status(); }

    /** The sequence inside the stored @p value. */
    sz::string_view_t sequence_of(value_type_ const& value) const noexcept {
        return {value.data() + sketch_bytes_, value.size() - sketch_bytes_};
    }

    void reset(value_type_ const& value) noexcept { canonical_.reset(sequence_of(value)); }
    void reset(std::string_view query) noexcept { canonical_.reset(query); }
    void reset(std::span<query_view_t const> queries) noexcept { canonical_.reset(queries); }
    basic::metric_staging_t push(value_type_ const& value) noexcept { return canonical_.push(sequence_of(value)); }
    void drain(std::span<distance_t> distances, distance_t radius) noexcept { canonical_.drain(distances, radius); }
    query_view_t query_view(value_type_ const& value) const noexcept { return query_view_t {sequence_of(value)}; }
    query_view_t query_view(std::string_view query) const noexcept { return metric_type_::query_view(query); }

  private:
    metric_type_ canonical_;
    std::size_t sketch_bytes_ = 0;
};

/**
 *  @brief The metric over a sketching kit's sketches: @c reset keeps the query's sketch, @c push
 *      stages a candidate's, and @c drain scores each pair by the kit's bound verb.
 *
 *  It reads the sketch of the index's bytes in front of each stored value and nothing of its
 *  sequence, so it holds no engine and no arena; a batch of queries is a span of sketches. Builders
 *  score by it, and a search's seeding walk scores the sketch @c sketch_of writes for its query.
 *
 *  @tparam kit_type_ The kit whose @c sketch_distance scores two sketches.
 *  @tparam value_type_ The stored value, its sketch in front.
 *  @tparam allocator_type_ places the staging and the query's sketch, one set per metric.
 *  @tparam cache_hints_type_ prefetches a staged candidate's sketch, ForkUnion's no-op by default.
 */
template <a_sketching_kit kit_type_, typename value_type_, typename allocator_type_ = std::allocator<byte_t>,
          typename cache_hints_type_ = fu::standard_cache_hints_t>
class stored_sketch_metric {
  public:
    using kit_t = kit_type_;
    using distance_t = sz_size_t;
    using allocator_t = allocator_type_;
    using sketch_tables_t = typename kit_type_::sketch_tables_t;
    using query_view_t = typename kit_type_::sketch_scalar_t const*;
    using is_transparent = void;

    stored_sketch_metric(allocator_t const& allocator = {}) noexcept
        : staged_(allocator), queries_(allocator), sketched_(allocator) {}

    /**
     *  @brief Sizes the staging for rounds of @p staging_limit candidates over sketches of
     *      @p sketch_bytes bytes, and the one sketch @ref sketch_of writes under @p tables with the
     *      scratch the kit states behind it.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k @p staging_limit or @p sketch_bytes is zero, or @p staging_limit
     *    is past @c u32_t slots.
     */
    status_t init(std::size_t staging_limit, std::size_t sketch_bytes, sketch_tables_t const& tables) noexcept {
        if (!staging_limit || staging_limit > (std::numeric_limits<u32_t>::max)() || !sketch_bytes)
            return status_code_t::config_mismatch_k;
        sketch_bytes_ = sketch_bytes, tables_ = tables;
        last_status_ = staged_.reserve_exactly(staging_limit);
        if (succeeded(last_status_))
            last_status_ = sketched_.resize(sketch_bytes + kit_t::sketch_scratch_bytes(sketch_bytes));
        return last_status_;
    }
    status_t status() const noexcept { return last_status_; }

    /** The sketch in front of the stored @p value. */
    query_view_t query_view(value_type_ const& value) const noexcept {
        return reinterpret_cast<query_view_t>(value.data());
    }

    /** Writes the sketch of a query @p stored in the form the index stores its sequences into the
     *  buffer @c init sized, valid until the next call. */
    query_view_t sketch_of(std::string_view stored) noexcept {
        bytes_span_t const sketched = to_writable_span(sketched_);
        kit_t::sketch_write(tables_, bytes_of(stored.data(), stored.size()), sketched.first(sketch_bytes_),
                            sketched.subspan(sketch_bytes_));
        return reinterpret_cast<query_view_t>(sketched_.data());
    }

    /** Opens a round for the sketch of @p value, discarding whatever the previous round staged. */
    void reset(value_type_ const& value) noexcept { reset(query_view(value)); }

    /** Opens a round for the sketch @p query, discarding whatever the previous round staged. */
    void reset(query_view_t query) noexcept {
        staged_.clear(), queries_.clear();
        last_status_ = queries_.reserve(1);
        if (succeeded(last_status_)) queries_.push_back_reserved(query);
    }

    /** Opens rounds scoring every sketch @p queries holds at once. */
    void reset(std::span<query_view_t const> queries) noexcept {
        staged_.clear(), queries_.clear();
        last_status_ = queries_.reserve(queries.size());
        if (failed(last_status_)) return;
        for (query_view_t const query : queries) queries_.push_back_reserved(query);
    }

    /** Stages the sketch of one candidate, or refuses once the round is full. */
    basic::metric_staging_t push(value_type_ const& value) noexcept {
        if (staged_.size() == staged_.capacity()) return basic::metric_staging_t::refused_k;
        query_view_t const sketch = query_view(value);
        for (std::size_t offset = 0; offset < sketch_bytes_; offset += fu::default_alignment_k)
            cache_hints_type_ {}(sketch + offset, fu::prefetch_line_k);
        staged_.push_back_reserved(sketch);
        return basic::metric_staging_t::staged_k;
    }

    /** Writes @b [queries,staged] distances, every one exact whatever the radius. */
    void drain(std::span<sz_size_t> distances, sz_size_t) noexcept {
        std::size_t const staged = staged_.size();
        for (std::size_t query = 0; query != queries_.size(); ++query)
            for (std::size_t candidate = 0; candidate != staged; ++candidate) {
                nk_u32_t distance = 0;
                [[maybe_unused]] nk_status_t const scored = kit_t::sketch_distance(queries_[query], staged_[candidate],
                                                                                   sketch_bytes_, &distance, nullptr);
                distances[query * staged + candidate] = distance;
            }
        staged_.clear();
    }

  private:
    growing_buffer<query_view_t, allocator_t> staged_;
    growing_buffer<query_view_t, allocator_t> queries_;

    /** The sketch @ref sketch_of wrote last, then the kit's scratch. */
    growing_buffer<byte_t, allocator_t> sketched_;
    sketch_tables_t tables_ {};
    std::size_t sketch_bytes_ = 0;
    status_t last_status_ = status_code_t::uninitialized_k;
};

/**
 *  @brief The seeding metric of a search over sequences stored without sketches, as an NVMe view
 *      reads them: @c push sketches each fetched sequence into one of the staging slots @c init
 *      carved, and @c drain scores every slot against the query's sketch by the kit's bound verb.
 *
 *  Its memory is the staging and the query's sketch, whatever the index holds.
 *
 *  @tparam kit_type_ The kit whose @c sketch_write and @c sketch_distance it calls.
 *  @tparam allocator_type_ places the slots, one set per metric.
 */
template <a_sketching_kit kit_type_, typename allocator_type_ = std::allocator<byte_t>>
class fetched_sketch_metric {
  public:
    using kit_t = kit_type_;
    using distance_t = sz_size_t;
    using allocator_t = allocator_type_;
    using sketch_tables_t = typename kit_type_::sketch_tables_t;
    using query_view_t = typename kit_type_::sketch_scalar_t const*;
    using is_transparent = void;

    fetched_sketch_metric(allocator_t const& allocator = {}) noexcept : sketches_(allocator) {}

    /**
     *  @brief Carves @p staging_limit slots and the query's sketch, each of @p sketch_bytes, all
     *      written under @p tables, and the scratch the kit states behind them.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k @p staging_limit or @p sketch_bytes is zero, or the slots overflow.
     */
    status_t init(std::size_t staging_limit, std::size_t sketch_bytes, sketch_tables_t const& tables) noexcept {
        std::size_t const scratch_bytes = kit_t::sketch_scratch_bytes(sketch_bytes);
        if (!staging_limit || !sketch_bytes ||
            staging_limit >= ((std::numeric_limits<std::size_t>::max)() - scratch_bytes) / sketch_bytes - 1)
            return status_code_t::config_mismatch_k;
        staging_limit_ = staging_limit, sketch_bytes_ = sketch_bytes, tables_ = tables;
        last_status_ = sketches_.resize((staging_limit + 1) * sketch_bytes + scratch_bytes);
        return last_status_;
    }
    status_t status() const noexcept { return last_status_; }

    /** Writes the sketch of a query @p stored in the form the index stores its sequences into the
     *  slot past the staging, valid until the next call. */
    query_view_t sketch_of(std::string_view stored) noexcept {
        bytes_span_t const sketch = slot_(staging_limit_);
        kit_t::sketch_write(tables_, bytes_of(stored.data(), stored.size()), sketch, scratch_());
        return reinterpret_cast<query_view_t>(sketch.data());
    }

    /** Opens a round for the sketch @p query, discarding whatever the previous round staged. */
    void reset(query_view_t query) noexcept { query_ = query, staged_ = 0; }

    /** Sketches one fetched @p sequence into the next slot, or refuses once no slot is free. */
    basic::metric_staging_t push(std::string_view sequence) noexcept {
        if (staged_ == staging_limit_) return basic::metric_staging_t::refused_k;
        kit_t::sketch_write(tables_, bytes_of(sequence.data(), sequence.size()), slot_(staged_), scratch_());
        ++staged_;
        return basic::metric_staging_t::staged_k;
    }

    /** Writes one distance per staged slot, every one exact whatever the radius. */
    void drain(std::span<sz_size_t> distances, sz_size_t) noexcept {
        for (std::size_t slot = 0; slot != staged_; ++slot) {
            nk_u32_t distance = 0;
            [[maybe_unused]] nk_status_t const scored = kit_t::sketch_distance(
                query_, reinterpret_cast<query_view_t>(slot_(slot).data()), sketch_bytes_, &distance, nullptr);
            distances[slot] = distance;
        }
        staged_ = 0;
    }

  private:
    /** Sketch slot @p slot, the query's past the staging. */
    bytes_span_t slot_(std::size_t slot) noexcept { return {sketches_.data() + slot * sketch_bytes_, sketch_bytes_}; }

    /** The kit's scratch past every slot. */
    bytes_span_t scratch_() noexcept {
        return to_writable_span(sketches_).subspan((staging_limit_ + 1) * sketch_bytes_);
    }

    /** The staged sketches, then the query's, then the kit's scratch. */
    growing_buffer<byte_t, allocator_t> sketches_;
    sketch_tables_t tables_ {};
    query_view_t query_ = nullptr;
    std::size_t staging_limit_ = 0;
    std::size_t sketch_bytes_ = 0;
    std::size_t staged_ = 0;
    status_t last_status_ = status_code_t::uninitialized_k;
};

/** The codec of a RAM index of @p kit_type_ : saves the sequence alone, as @ref codec_t saves it,
 *  and loads it behind the sketch of @c sketch_bytes a sketching kit writes in place from
 *  @c tables, through the scratch the kit states. */
template <typename kit_type_>
struct stored_codec {
    inline static constexpr dtype_t value_dtype_k = dtype_t::utf8_k;

    /** The tables the kit's sketch reads, borrowed from the variant, which outlives every plan. */
    typename kit_type_::sketch_tables_t tables {};

    /** Bytes of the sketch in front of every stored sequence; zero for a kit that writes none. */
    std::size_t sketch_bytes = 0;

    template <typename string_type_>
    std::size_t storage_requirements(string_type_ const& value) const noexcept {
        return value.size() - sketch_bytes;
    }

    /**
     *  @brief Copies the sequence of @p value into @p output as it is.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k @p output differs in length from the sequence.
     */
    template <typename string_type_>
    status_t save(string_type_ const& value, bytes_span_t output) const noexcept {
        return copy_bytes(output, std::as_bytes(std::span {value}).subspan(sketch_bytes));
    }

    /** Bytes of the scratch each @ref load sketches through; zero for a kit that writes none. */
    std::size_t scratch_bytes() const noexcept {
        if constexpr (a_sketching_kit<kit_type_>) return kit_type_::sketch_scratch_bytes(sketch_bytes);
        else return 0;
    }

    /** Writes @p input into @p value behind its sketch, through the string's own allocator and
     *  @p scratch of @ref scratch_bytes. */
    template <typename string_type_>
    status_t load(bytes_view_t input, string_type_& value, bytes_span_t scratch) const noexcept {
        std::size_t const bytes = sketch_bytes + input.size();
        return status_of(value.resize_and_overwrite(bytes, [&](char* stored, std::size_t) noexcept {
            bytes_span_t const sequence = bytes_of(stored + sketch_bytes, input.size());
            if (!input.empty()) std::memcpy(sequence.data(), input.data(), input.size());
            if constexpr (a_sketching_kit<kit_type_>)
                kit_type_::sketch_write(tables, sequence, bytes_of(stored, sketch_bytes), scratch);
            return bytes;
        }));
    }
};

#pragma endregion Stored Values

} // namespace unum::usearch::sequence
