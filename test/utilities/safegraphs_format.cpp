/**
 *  @brief The interchange unit of the suite: a fixed-width embedding fixture carried through
 *      the shared basic-tier SafeGraphs scenarios - monolith, hostile headers, and the
 *      distributed manifest form.
 *
 *  @author Ash Vardanian
 *  @file test/utilities/safegraphs_format.cpp
 *  @date September 3, 2026
 *
 *  Every modality unit brings a ragged value column, so a fixed-width one is what this fixture
 *  adds: its header must say @c FixedSizeList where theirs say a prefix-sum list, and the same
 *  scenarios must carry it end to end.
 */
#include <cmath> // `std::sqrt` - the angular oracle

#include <array>  // `std::array` - the fixed embedding
#include <bit>    // `std::bit_cast` - byte images of the fixed payload
#include <span>   // `std::span` - the metric's drain surface
#include <tuple>  // `std::tuple_size_v` - the embedding's width
#include <vector> // `std::vector` - the staged batch & the generated values

#include "basic.hpp"

#include "usearch/utilities/safegraphs_format.hpp"

using namespace unum::usearch;

#pragma region Embeddings

/** @brief One fixed-width vector; plain floats, so the codec round trip is bit-exact. */
using embedding_t = std::array<f32_t, 16>;

/** @brief One minus the cosine - the reference's @c angular over unit-ish vectors. */
static f32_t angular_distance(embedding_t const& first, embedding_t const& second) noexcept {
    f32_t dot = 0, first_norm = 0, second_norm = 0;
    for (std::size_t axis = 0; axis != first.size(); ++axis) {
        dot += first[axis] * second[axis];
        first_norm += first[axis] * first[axis];
        second_norm += second[axis] * second[axis];
    }
    return 1.0f - dot / std::sqrt(first_norm * second_norm);
}

/** @brief The batched metric over stored embeddings, staged by value - a fixed row of floats
 *      is cheaper copied than chased. */
struct angular_metric_t {
    embedding_t query_ {};
    std::vector<embedding_t> staged_;

    void reset(embedding_t const& query) noexcept { query_ = query, staged_.clear(); }
    basic_metric_staging_t push(embedding_t const& value) noexcept {
        return staged_.push_back(value), basic_metric_staging_t::staged_k;
    }
    void drain(std::span<f32_t> distances) noexcept {
        for (std::size_t lane = 0; lane != distances.size(); ++lane)
            distances[lane] = angular_distance(query_, staged_[lane]);
        staged_.clear();
    }
};

/** @brief The embedding's own bytes: one fixed row of sixteen @c f32, a regular column. */
struct embedding_codec_t {
    inline static constexpr dtype_t value_dtype_k = dtype_t::f32_k;
    inline static constexpr column_shape_t value_shape_k = vector_shape_t {std::tuple_size_v<embedding_t>};

    std::size_t storage_requirements(embedding_t const&) const noexcept { return sizeof(embedding_t); }
    status_t save(embedding_t const& value, bytes_span_t output) const noexcept {
        if (output.size() < sizeof(embedding_t)) return status_t::overflow_k;
        auto const image = std::bit_cast<std::array<std::byte, sizeof(embedding_t)>>(value);
        std::copy_n(image.data(), image.size(), output.data());
        return status_t::success_k;
    }
    status_t load(bytes_view_t input, embedding_t& value) const noexcept {
        if (input.size() != sizeof(embedding_t)) return status_t::malformed_k;
        std::array<std::byte, sizeof(embedding_t)> image;
        std::copy_n(input.data(), image.size(), image.data());
        value = std::bit_cast<embedding_t>(image);
        return status_t::success_k;
    }
};

struct embeddings_fixture_t {
    using index_t =
        basic_ram_index<embedding_t, u64_t, u32_t, f32_t, dummy_waiter_t, std::atomic_ref, counting_allocator_t>;
    static constexpr char const* metric_k = "angular";
    static constexpr char const* key_dtype_k = "u64";

    static index_t make_index(allocation_ledger_t& ledger) { return index_t {counting_allocator_t {ledger}}; }
    static angular_metric_t make_metric() { return {}; }
    static embedding_codec_t make_codec() { return {}; }
    static u64_t make_key(std::size_t ordinal) { return 0x5EED000000000000ull + ordinal; }
    static bool value_equals(embedding_t const& first, embedding_t const& second) noexcept { return first == second; }

    /** @brief A few prototypes on the unit sphere, each drawn many times with a small jitter,
     *      so the graph has clusters to navigate. */
    static std::vector<embedding_t> make_values(std::size_t count, u64_t seed) {
        constexpr std::size_t prototypes_k = 6;
        splitmix64_t random_generator {seed};
        uniform_below<u32_t> coordinate_distribution {2001};
        uniform_below<u32_t> prototype_distribution {prototypes_k};
        auto draw_signed = [&](f32_t amplitude) noexcept {
            return (static_cast<f32_t>(coordinate_distribution(random_generator)) / 1000.0f - 1.0f) * amplitude;
        };
        std::array<embedding_t, prototypes_k> prototypes;
        for (embedding_t& prototype : prototypes)
            for (f32_t& coordinate : prototype) coordinate = draw_signed(1);
        std::vector<embedding_t> values;
        for (std::size_t ordinal = 0; ordinal != count; ++ordinal) {
            embedding_t value = prototypes[prototype_distribution(random_generator)];
            for (f32_t& coordinate : value) coordinate += draw_signed(0.05f);
            values.push_back(value);
        }
        return values;
    }
};

static_assert(basic_value<embedding_t>);
static_assert(basic_metric<angular_metric_t, embedding_t, f32_t>);
static_assert(basic_value_codec<embedding_codec_t, embedding_t>);
static_assert(interchange_value_codec<embedding_codec_t>);

#pragma endregion Embeddings

#pragma region Harness

/** @brief The shared basic-tier scenarios over the embeddings fixture - a fixed-width value
 *      column, which no modality unit has: monolith, hostile header battery, and the
 *      distributed form, each at every alignment and executor the scenarios cover. */
std::size_t test_safegraphs_format() {
    embeddings_fixture_t const fixture;
    std::size_t failures = 0;
    failures += test_basic_safegraphs_roundtrip(fixture, "u64", embeddings_fixture_t::metric_k);
    failures += test_basic_safegraphs_hostile(fixture, embeddings_fixture_t::metric_k);
    failures += test_basic_safegraphs_distributed(fixture, embeddings_fixture_t::metric_k);
    return report("safegraphs format", failures);
}

#pragma endregion Harness
