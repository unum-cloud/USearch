/**
 *  @file test/utilities/safegraphs_format.cpp
 *  @author Ash Vardanian
 *  @date September 3, 2026
 *  @brief The interchange unit of the suite: a fixed-width embedding fixture carried through the
 *      shared basic-tier SafeGraphs scenarios - monolith, hostile headers, and the distributed
 *      manifest form.
 *
 *  Every modality unit brings a ragged value column, so a fixed-width one is what this fixture
 *  adds: its header must say @c FixedSizeList where theirs say a prefix-sum list, and the same
 *  scenarios must carry it end to end.
 */
#include <cmath>   // `std::sqrt` - the angular oracle
#include <cstdio>  // `std::snprintf`, `std::remove` - the saved monoliths
#include <cstring> // `std::memcpy` - the codec's views of stored entries

#include <algorithm> // `std::find` - a query's saved keys
#include <array>     // `std::array` - the fixed embedding
#include <bit>       // `std::bit_cast` - byte images of the fixed payload
#include <span>      // `std::span` - the metric's drain surface
#include <tuple>     // `std::tuple_size_v` - the embedding's width
#include <vector>    // `std::vector` - the staged batch & the generated values

#include "basic.hpp"

#include "usearch/basic/nvme_indexes.hpp" // `basic::nvme_indexes` - the UUID-keyed view on storage
#include "usearch/utilities/safegraphs_format.hpp"

using namespace unum::usearch;
using namespace unum::usearch::test;

#pragma region Embeddings

/** One fixed-width vector; plain floats, so the codec round trip is bit-exact. */
using embedding_t = std::array<f32_t, 16>;

/** One minus the cosine - the reference's @c angular over unit-ish vectors. */
static f32_t angular_distance(embedding_t const& first, embedding_t const& second) noexcept {
    f32_t dot = 0, first_norm = 0, second_norm = 0;
    for (std::size_t axis = 0; axis != first.size(); ++axis) {
        dot += first[axis] * second[axis];
        first_norm += first[axis] * first[axis];
        second_norm += second[axis] * second[axis];
    }
    return 1.0f - dot / std::sqrt(first_norm * second_norm);
}

/** The batched metric over stored embeddings, staged by value - a fixed run of floats
 *  is cheaper copied than chased. */
struct angular_metric_t {
    embedding_t query_ {};
    std::vector<embedding_t> staged_;

    void reset(embedding_t const& query) noexcept { query_ = query, staged_.clear(); }
    metric_staging_t push(embedding_t const& value) noexcept {
        return staged_.push_back(value), metric_staging_t::staged_k;
    }
    void drain(std::span<f32_t> distances, f32_t) noexcept {
        for (std::size_t lane = 0; lane != distances.size(); ++lane)
            distances[lane] = angular_distance(query_, staged_[lane]);
        staged_.clear();
    }
};

/** The embedding's own bytes: one fixed entry of sixteen @c f32, a regular column. */
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

    /** What @ref view hands out for a stored entry: the embedding itself, copied. */
    using view_t = embedding_t;

    /** The embedding a stored @p entry holds. */
    view_t view(bytes_view_t entry) const noexcept {
        embedding_t value {};
        if (entry.size() == sizeof(embedding_t)) std::memcpy(value.data(), entry.data(), sizeof(embedding_t));
        return value;
    }
};

/** The embeddings over @p key_type_ keys and @p node_id_type_ node IDs. */
template <typename key_type_, typename node_id_type_>
struct embeddings_fixture {
    using index_t = ram_index<embedding_t, key_type_, node_id_type_, f32_t, dummy_waiter_t, fu::standard_atomic_ref,
                              counting_allocator_t>;

    static constexpr char const* metric_k = "angular";

    static std::unique_ptr<index_t> make_index(allocation_ledger_t& ledger) {
        return std::unique_ptr<index_t>(new (std::nothrow) index_t(counting_allocator_t {ledger}));
    }
    static angular_metric_t make_metric() { return {}; }
    static embedding_codec_t make_codec() { return {}; }
    static key_type_ make_key(std::size_t ordinal) {
        return static_cast<key_type_>((sizeof(key_type_) == 8 ? 0x5EED000000000000ull : 0x5EED0000ull) + ordinal);
    }
    static bool value_equals(embedding_t const& first, embedding_t const& second) noexcept { return first == second; }

    /** A few prototypes on the unit sphere, each drawn many times with a small jitter,
     *  so the graph has clusters to navigate. */
    static std::vector<embedding_t> make_values(std::size_t count, u64_t seed, typename index_t::allocator_t const&) {
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

using embeddings_fixture_t = embeddings_fixture<u64_t, u32_t>;

static_assert(value<embedding_t>);
static_assert(metric<angular_metric_t, embedding_t, f32_t>);
static_assert(value_codec<embedding_codec_t, embedding_t>);
static_assert(interchange_value_codec<embedding_codec_t>);

/** A UUID-keyed RAM index over the embeddings, the target every narrower key column widens into. */
using widened_index_t =
    ram_index<embedding_t, uuid_t, u32_t, f32_t, dummy_waiter_t, fu::standard_atomic_ref, counting_allocator_t>;

/** A UUID-keyed view of the embeddings on storage. */
using widened_view_t =
    us::basic::nvme_indexes<embedding_codec_t, uuid_t, u32_t, f32_t, blocking_files_engine<fu::standard_yield_t>>;

/**
 *  @brief A monolith of @p fixture_type_ 's keys widens into UUID keys on every reader.
 *
 *  Its keys read back as @c uuid_of renders them, big-endian in the trailing eight bytes, through a
 *  load into a UUID-keyed RAM index and through a UUID-keyed view on storage, and both rank every
 *  entry's nearest entries at the distances the saved index does.
 */
template <typename fixture_type_>
static std::size_t expect_keys_widen(u64_t seed, std::string_view label) {
    using source_t = typename fixture_type_::index_t;
    using node_id_t = typename source_t::node_id_t;
    constexpr std::size_t nodes_k = 24, wanted_k = 4;
    std::size_t failures = 0;
    allocation_ledger_t ledger;
    char path[64];
    std::snprintf(path, sizeof(path), "/tmp/usearch-widen-%08x.safegraphs",
                  static_cast<unsigned>(stream_key(seed, label) ^ reinterpret_cast<std::uintptr_t>(&ledger)));
    std::vector<embedding_t> values = fixture_type_::make_values(nodes_k, stream_key(seed, label),
                                                                 counting_allocator_t {ledger});
    std::vector<uuid_t> expected_keys(nodes_k * wanted_k);
    std::vector<f32_t> expected_distances(nodes_k * wanted_k);
    serial_executor_t serial;
    // A query ranks alike when its distances agree within noise and every key nearer than the
    // query's last distance is one the saved index ranked there too: ties at the cut may swap keys.
    error_stats_t distance_errors;
    auto ranked_alike = [&](std::size_t query, std::span<uuid_t const> keys, std::span<f32_t const> distances) {
        std::size_t mismatches = 0;
        std::span<uuid_t const> const wanted_keys = std::span<uuid_t const> {expected_keys}.subspan(query * wanted_k,
                                                                                                    wanted_k);
        f32_t const last = expected_distances[query * wanted_k + wanted_k - 1];
        for (std::size_t rank = 0; rank != wanted_k; ++rank) {
            distance_errors.add(distances[rank], expected_distances[query * wanted_k + rank]);
            bool const before_cut = distances[rank] < last * (1 - 1e-3f);
            mismatches += before_cut &&
                          std::find(wanted_keys.begin(), wanted_keys.end(), keys[rank]) == wanted_keys.end();
        }
        return mismatches;
    };
    {
        std::unique_ptr<source_t> const source_ptr = fixture_type_::make_index(ledger);
        source_t& source = *source_ptr;
        angular_metric_t metric;
        typename source_t::traversal_context_t context {stream_key(seed, label), counting_allocator_t {ledger}};
        std::vector<node_id_t> assigned_ids;
        failures += expect(bool(source.reserve(nodes_k)));
        failures += basic_build<fixture_type_>(source, metric, values, assigned_ids, context, keep_diverse_t {4});
        for (std::size_t query = 0; query != nodes_k; ++query) {
            failures += expect_eq(source.search(values[query], {nodes_k}, metric, context).status, status_t::success_k);
            for (std::size_t rank = 0; rank != wanted_k && rank < context.results.size(); ++rank) {
                auto const key = source.key_of(context.results[rank].node_id);
                expected_keys[query * wanted_k + rank] = uuid_of(static_cast<u64_t>(key.value));
                expected_distances[query * wanted_k + rank] = context.results[rank].distance;
            }
        }
        failures += expect_eq(safegraphs_write_index(std::string_view {path}, source, embedding_codec_t {}, serial,
                                                     save_config_t {}, embeddings_fixture_t::metric_k),
                              status_t::success_k);
    }

    // The RAM load widens every key and keeps the graph, so its walks rank alike.
    {
        auto [mapping, map_status] = mapped_file_t::read(path);
        failures += expect(succeeded(map_status));
        widened_index_t widened {counting_allocator_t {ledger}};
        failures += expect_eq(
            safegraphs_read_index(mapping.view(), widened, embedding_codec_t {}, serial, load_config_t {}),
            status_t::success_k);
        failures += expect_eq(widened.size(), nodes_k);
        angular_metric_t metric;
        widened_index_t::traversal_context_t context {stream_key(seed, label), counting_allocator_t {ledger}};
        for (std::size_t query = 0; query != nodes_k; ++query) {
            failures += expect_eq(widened.search(values[query], {nodes_k}, metric, context).status,
                                  status_t::success_k);
            failures += expect(context.results.size() >= wanted_k);
            if (context.results.size() < wanted_k) continue;
            std::array<uuid_t, wanted_k> keys;
            std::array<f32_t, wanted_k> distances;
            for (std::size_t rank = 0; rank != wanted_k; ++rank)
                keys[rank] = widened.key_of(context.results[rank].node_id).value,
                distances[rank] = context.results[rank].distance;
            failures += ranked_alike(query, keys, distances);
        }
    }

    // The view widens the same keys on storage: each counts once, and the walks rank alike.
    fu_topology_t topology = nullptr;
    failures += expect(fu_topology_new(&topology) == fu_success_k);
    {
        io_engine_config_t config;
        config.queue_depth = 64;
        auto [executor, spawn_status] = widened_view_t::executor_t::spawn(topology, 1, config);
        failures += expect_eq(spawn_status, status_t::success_k);
        widened_view_t viewed {forkunion_allocator_t {}};
        failures += expect_eq(viewed.view(path, {}, embeddings_fixture_t::metric_k, executor), status_t::success_k);
        failures += expect_eq(viewed.size(), nodes_k);
        std::vector<std::size_t> entries(expected_keys.size());
        failures += expect_eq(viewed.count(std::span<uuid_t const> {expected_keys}, std::span {entries}, executor),
                              status_t::success_k);
        for (std::size_t const counted : entries) failures += expect_eq(counted, 1u);
        auto make_metric = [](auto const&, std::size_t) noexcept {
            return expected<angular_metric_t> {angular_metric_t {}, status_t::success_k};
        };
        auto query_of = [&](std::size_t query, auto const&) noexcept {
            return expected<embedding_t> {values[query], status_t::success_k};
        };
        widened_view_t::search_config_t search_config;
        search_config.expansion_factor = nodes_k;
        std::vector<uuid_t> keys(nodes_k * wanted_k);
        std::vector<f32_t> distances(nodes_k * wanted_k);
        std::vector<std::size_t> counts(nodes_k);
        auto const [found, search_status] = viewed.search(nodes_k, make_metric, query_of, std::span<uuid_t> {keys},
                                                          std::span<f32_t> {distances}, std::span<std::size_t> {counts},
                                                          search_config, executor);
        failures += expect_eq(search_status, status_t::success_k);
        for (std::size_t query = 0; query != nodes_k; ++query)
            failures += ranked_alike(query, std::span<uuid_t const> {keys}.subspan(query * wanted_k, wanted_k),
                                     std::span<f32_t const> {distances}.subspan(query * wanted_k, wanted_k));
    }
    fu_topology_delete(topology);
    std::remove(path);
    failures += expect_errors("widened distances", comparison_family_t::approximate_k, comparison_verdict_t::assert_k,
                              distance_errors, 0);
    return failures + expect_eq(ledger.live_bytes, 0);
}

/** A monolith of 64-bit node IDs reaches no reader of 32-bit ones: the UUID-keyed RAM load and the
 *  UUID-keyed view both refuse it as a mismatched configuration. */
static std::size_t expect_wide_nodes_refused(u64_t seed) {
    using fixture_t = embeddings_fixture<u64_t, u64_t>;
    using source_t = fixture_t::index_t;
    constexpr std::size_t nodes_k = 16;
    std::size_t failures = 0;
    allocation_ledger_t ledger;
    char path[64];
    std::snprintf(path, sizeof(path), "/tmp/usearch-wide-nodes-%08x.safegraphs",
                  static_cast<unsigned>(stream_key(seed, "wide nodes") ^ reinterpret_cast<std::uintptr_t>(&ledger)));
    serial_executor_t serial;
    {
        std::unique_ptr<source_t> const source_ptr = fixture_t::make_index(ledger);
        source_t& source = *source_ptr;
        angular_metric_t metric;
        source_t::traversal_context_t context {stream_key(seed, "wide nodes"), counting_allocator_t {ledger}};
        std::vector<embedding_t> values = fixture_t::make_values(nodes_k, stream_key(seed, "wide nodes"),
                                                                 counting_allocator_t {ledger});
        std::vector<u64_t> assigned_ids;
        failures += expect(bool(source.reserve(nodes_k)));
        failures += basic_build<fixture_t>(source, metric, values, assigned_ids, context, keep_diverse_t {4});
        failures += expect_eq(safegraphs_write_index(std::string_view {path}, source, embedding_codec_t {}, serial,
                                                     save_config_t {}, embeddings_fixture_t::metric_k),
                              status_t::success_k);
    }
    {
        auto [mapping, map_status] = mapped_file_t::read(path);
        failures += expect(succeeded(map_status));
        widened_index_t widened {counting_allocator_t {ledger}};
        failures += expect_eq(
            safegraphs_read_index(mapping.view(), widened, embedding_codec_t {}, serial, load_config_t {}),
            status_t::config_mismatch_k);
    }
    fu_topology_t topology = nullptr;
    failures += expect(fu_topology_new(&topology) == fu_success_k);
    {
        io_engine_config_t config;
        config.queue_depth = 64;
        auto [executor, spawn_status] = widened_view_t::executor_t::spawn(topology, 1, config);
        failures += expect_eq(spawn_status, status_t::success_k);
        widened_view_t viewed {forkunion_allocator_t {}};
        failures += expect_eq(viewed.view(path, {}, embeddings_fixture_t::metric_k, executor),
                              status_t::config_mismatch_k);
        // The little-endian header length follows the magic; 2^63 bytes is past any file.
        unsigned char const header_length[8] = {0, 0, 0, 0, 0, 0, 0, 0x80};
        std::FILE* const corrupted = std::fopen(path, "r+b");
        failures += expect(corrupted != nullptr);
        failures += expect(std::fseek(corrupted, 8, SEEK_SET) == 0 && std::fwrite(header_length, 1, 8, corrupted) == 8);
        std::fclose(corrupted);
        widened_view_t corrupted_view {forkunion_allocator_t {}};
        failures += expect_eq(corrupted_view.view(path, {}, embeddings_fixture_t::metric_k, executor),
                              status_t::malformed_k);
    }
    fu_topology_delete(topology);
    std::remove(path);
    return failures + expect_eq(ledger.live_bytes, 0);
}

/**
 *  @brief A checkpoint read as if a device held it crosses into host memory only through the
 *      caller's copy, and only as its manifest and headers, while every part keeps viewing the
 *      source; a column then copies through the same callback.
 *
 *  The copy records what it read, from the source's start: a monolith reads nothing past its
 *  data offset, and a two-part buffer nothing but its manifest and each part's header.
 */
static std::size_t expect_view_reads_headers_alone(u64_t seed) {
    using index_t = embeddings_fixture_t::index_t;
    constexpr std::size_t nodes_k = 24, parts_k = 2;
    std::size_t failures = 0;
    allocation_ledger_t ledger;
    std::vector<byte_t> monolith;
    {
        serial_executor_t serial;
        std::unique_ptr<index_t> const index = embeddings_fixture_t::make_index(ledger);
        angular_metric_t metric;
        index_t::traversal_context_t context {stream_key(seed, "header alone"), counting_allocator_t {ledger}};
        std::vector<embedding_t> values = embeddings_fixture_t::make_values(nodes_k, stream_key(seed, "header alone"),
                                                                            counting_allocator_t {ledger});
        std::vector<u32_t> assigned_ids;
        failures += expect(bool(index->reserve(nodes_k)));
        failures += basic_build<embeddings_fixture_t>(*index, metric, values, assigned_ids, context,
                                                      keep_diverse_t {4});
        auto const [bytes, measure_status] = safegraphs_storage_requirements(
            *index, embedding_codec_t {}, save_config_t {}, embeddings_fixture_t::metric_k);
        failures += expect(succeeded(measure_status));
        monolith.resize(bytes);
        failures += expect_eq(safegraphs_write_index(bytes_span_t {monolith}, *index, embedding_codec_t {}, serial,
                                                     save_config_t {}, embeddings_fixture_t::metric_k),
                              status_t::success_k);
    }

    byte_t const* origin = nullptr;
    std::vector<iota_view<std::size_t>> crossed;
    auto device_copy = [&](bytes_span_t host, bytes_view_t device) noexcept {
        std::size_t const first = static_cast<std::size_t>(device.data() - origin);
        crossed.push_back({first, first + device.size()});
        return copy_bytes(host, device);
    };
    using parts_t = growing_buffer<safegraphs_catalog_part_t, counting_allocator_t>;
    using extra_columns_t = growing_buffer<safegraphs_named_column_t, counting_allocator_t>;

    // A monolith: its header alone crosses, then its key column copies through the callback.
    origin = monolith.data();
    {
        checkpoint_source_t const source {bytes_view_t {monolith}, 0};
        parts_t parts {counting_allocator_t {ledger}};
        extra_columns_t extra_columns {counting_allocator_t {ledger}};
        failures += expect_eq(source.read_header(parts, extra_columns, device_copy, counting_allocator_t {ledger}),
                              status_t::success_k);
        failures += expect_eq(parts.size(), 1u);
        if (parts.size() != 1) return failures;
        safegraphs_header_t const& header = parts[0].header;
        failures += expect_eq(parts[0].nodes, nodes_k);
        failures += expect(parts[0].monolith.data() == monolith.data());
        failures += expect(parts[0].metric.view() == embeddings_fixture_t::metric_k);
        for (iota_view<std::size_t> const range : crossed) failures += expect(range.bound <= header.data_offset);

        crossed.clear();
        std::vector<byte_t> keys(nodes_k * header.key.row_bytes()), host_keys(keys.size());
        failures += expect_eq(source.copy_column(header, header.key, {0, nodes_k}, bytes_span_t {keys}, device_copy),
                              status_t::success_k);
        failures += expect_eq(crossed.size(), 1u);
        // Host bytes copy without the callback, which this one would refuse.
        checkpoint_source_t const in_place {bytes_view_t {monolith}};
        failures += expect_eq(
            in_place.copy_column(header, header.key, {0, nodes_k}, bytes_span_t {host_keys}, no_device_copy_t {}),
            status_t::success_k);
        failures += expect(keys == host_keys);
    }

    // Two copies of it behind a manifest leading one buffer: the manifest and each header cross.
    std::array<std::size_t, parts_k> offsets {};
    auto part_at = [&](std::size_t part) noexcept {
        return safegraphs_part_t {{}, monolith.size(), nodes_k, offsets[part]};
    };
    auto const [buffer_bytes, layout_status] = safegraphs_buffer_layout(
        parts_k, part_at, safegraphs_alignment_default_k, std::span<std::size_t> {offsets});
    failures += expect_eq(layout_status, status_t::success_k);
    std::vector<byte_t> buffer(buffer_bytes);
    failures += expect_eq(safegraphs_write_manifest(bytes_span_t {buffer}, parts_k, part_at), status_t::success_k);
    for (std::size_t const offset : offsets) std::copy(monolith.begin(), monolith.end(), buffer.begin() + offset);
    origin = buffer.data();
    crossed.clear();
    {
        checkpoint_source_t const source {bytes_view_t {buffer}, 0};
        parts_t parts {counting_allocator_t {ledger}};
        extra_columns_t extra_columns {counting_allocator_t {ledger}};
        failures += expect_eq(source.read_header(parts, extra_columns, device_copy, counting_allocator_t {ledger}),
                              status_t::success_k);
        failures += expect_eq(parts.size(), parts_k);
        if (parts.size() != parts_k) return failures;
        for (std::size_t part = 0; part != parts_k; ++part)
            failures += expect(parts[part].monolith.data() == buffer.data() + offsets[part]);
        for (iota_view<std::size_t> const range : crossed) {
            bool inside = range.bound <= offsets[0];
            for (std::size_t part = 0; part != parts_k; ++part)
                inside |= range.first >= offsets[part] && range.bound <= offsets[part] + parts[part].header.data_offset;
            failures += expect(inside);
        }
        auto const [metadata, metadata_status] = safegraphs_read_metadata(source, device_copy,
                                                                          counting_allocator_t {ledger});
        failures += expect_eq(metadata_status, status_t::success_k);
        failures += expect_eq(metadata.parts, parts_k);
        failures += expect_eq(metadata.nodes, parts_k * nodes_k);
    }
    return failures + expect_eq(ledger.live_bytes, 0);
}

/** The shared basic-tier scenarios over the embeddings fixture - a fixed-width value column, which
 *  no modality unit has: monolith, hostile header battery, and the distributed form, each at every
 *  alignment and executor the scenarios cover - then the narrower key columns widening into UUIDs,
 *  the wider node IDs refused, and a device's checkpoint read through its headers alone. */
std::size_t test::utilities::check_safegraphs_format(settings_t const& settings) {
    std::size_t failures = 0;
    failures += test_basic_safegraphs_roundtrip<embeddings_fixture_t>(settings.seed, "u64",
                                                                      embeddings_fixture_t::metric_k);
    failures += test_basic_safegraphs_hostile<embeddings_fixture_t>(settings.seed, embeddings_fixture_t::metric_k);
    failures += test_basic_safegraphs_distributed<embeddings_fixture_t>(settings.seed, embeddings_fixture_t::metric_k);
    failures += expect_keys_widen<embeddings_fixture_t>(settings.seed, "u64 keys");
    failures += expect_keys_widen<embeddings_fixture<u32_t, u32_t>>(settings.seed, "u32 keys");
    failures += expect_wide_nodes_refused(settings.seed);
    failures += expect_view_reads_headers_alone(settings.seed);
    return report("safegraphs format", failures);
}

#pragma endregion Embeddings
