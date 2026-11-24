/**
 *  @file test/utilities/safegraphs_format.cpp
 *  @author Ash Vardanian
 *  @date September 3, 2026
 *  @brief The interchange unit of the suite: a fixed-width embedding fixture held through the
 *      shared basic-tier SafeGraphs scenarios - monolith, hostile headers, and the distributed
 *      manifest form.
 *
 *  Every modality unit brings a ragged value column, so a fixed-width one is what this fixture
 *  adds: its header must say @c FixedSizeList where theirs say a prefix-sum list, and the same
 *  scenarios must hold it end to end.
 */
#undef NDEBUG // ! A test's oracle must stay live in every build

#include <cmath>   // `std::sqrt` - the angular oracle
#include <cstdio>  // `std::fopen`, `std::fseek`, `std::fwrite` - the corrupted header
#include <cstring> // `std::memcpy`, `std::memcmp` - the codec's views of stored entries, read rows

#include <algorithm> // `std::find` - a query's saved keys
#include <array>     // `std::array` - the fixed embedding
#include <bit>       // `std::bit_cast` - byte images of the fixed payload
#include <span>      // `std::span` - the metric's drain surface
#include <tuple>     // `std::tuple_size_v` - the embedding's width
#include <vector>    // `std::vector` - the staged batch & the generated values

#include <fmt/format.h> // `fmt::format` - the address inside a device's span

#include "basic.hpp"

#include "usearch/basic/nvme_indexes.hpp" // `basic::nvme_indexes` - the UUID-keyed view on storage
#include "usearch/utilities/safegraphs_format.hpp"

namespace util = unum::usearch::utilities;

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
        if (output.size() < sizeof(embedding_t)) return status_code_t::overflow_k;
        auto const image = std::bit_cast<std::array<std::byte, sizeof(embedding_t)>>(value);
        std::copy_n(image.data(), image.size(), output.data());
        return status_code_t::success_k;
    }
    status_t load(bytes_view_t input, embedding_t& value) const noexcept {
        if (input.size() != sizeof(embedding_t)) return status_code_t::malformed_k;
        std::array<std::byte, sizeof(embedding_t)> image;
        std::copy_n(input.data(), image.size(), image.data());
        value = std::bit_cast<embedding_t>(image);
        return status_code_t::success_k;
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
        split_mix_generator_t random_generator {seed};
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
using widened_index_t = ram_index<embedding_t, util::uuid128_t, u32_t, f32_t, dummy_waiter_t, fu::standard_atomic_ref,
                                  counting_allocator_t>;

/** A UUID-keyed view of the embeddings on storage. */
using widened_view_t = us::basic::nvme_indexes<embedding_codec_t, util::uuid128_t, u32_t, f32_t,
                                               blocking_files_queue<fu::standard_yield_t>>;

/**
 *  @brief A monolith of @p fixture_type_ 's keys widens into UUID keys on every reader.
 *
 *  Its keys read back as @c uuid_of renders them, big-endian in the trailing eight bytes, through a
 *  load into a UUID-keyed RAM index and through a UUID-keyed view on storage, and both rank every
 *  entry's nearest entries at the distances the saved index does.
 */
template <typename fixture_type_>
static void test_keys_widen(u64_t seed, std::string_view label) {
    using source_t = typename fixture_type_::index_t;
    using node_id_t = typename source_t::node_id_t;
    constexpr std::size_t nodes_k = 24, wanted_k = 4;
    allocation_ledger_t ledger;
    scratch_directory_t const scratch;
    std::string const path = scratch.path("widen.safegraphs");
    std::vector<embedding_t> values = fixture_type_::make_values(nodes_k, stream_key(seed, label),
                                                                 counting_allocator_t {ledger});
    nk::matrix<util::uuid128_t> expected_keys = make_matrix<util::uuid128_t>(nodes_k, wanted_k);
    nk::matrix<f32_t> expected_distances = make_matrix<f32_t>(nodes_k, wanted_k);
    serial_executor_t serial;
    // A query ranks alike when its distances agree within noise and every key nearer than the
    // query's last distance is one the saved index ranked there too: ties at the cut may swap keys.
    error_stats_t distance_errors;
    auto ranked_alike = [&](std::size_t query, std::span<util::uuid128_t const> keys,
                            std::span<f32_t const> distances) {
        std::size_t mismatches = 0;
        std::span<util::uuid128_t const> const wanted_keys = to_span(expected_keys.view().row(query).as_vector());
        f32_t const last = expected_distances(query, wanted_k - 1);
        for (std::size_t rank = 0; rank != wanted_k; ++rank) {
            distance_errors.add(distances[rank], expected_distances(query, rank));
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
        expect_true(bool(source.reserve(nodes_k)));
        basic_build<fixture_type_>(source, metric, values, assigned_ids, context, keep_diverse_t {4});
        for (std::size_t query = 0; query != nodes_k; ++query) {
            expect_equal(source.search(values[query], {nodes_k}, metric, context).status, status_code_t::success_k);
            for (std::size_t rank = 0; rank != wanted_k && rank < context.results.size(); ++rank) {
                auto const key = source.key_of(context.results[rank].node_id);
                expected_keys(query, rank) = uuid_of(static_cast<u64_t>(key.value));
                expected_distances(query, rank) = context.results[rank].distance;
            }
        }
        expect_equal(safegraphs_write_index(std::string_view {path}, source, embedding_codec_t {}, serial,
                                            save_config_t {}, embeddings_fixture_t::metric_k),
                     status_code_t::success_k);
    }

    // The RAM load widens every key and keeps the graph, so its walks rank alike.
    {
        auto [mapping, map_status] = mapped_file_t::read(path);
        expect_true(succeeded(map_status));
        widened_index_t widened {counting_allocator_t {ledger}};
        expect_equal(safegraphs_read_index(mapping.view(), widened, embedding_codec_t {}, serial, load_config_t {}),
                     status_code_t::success_k);
        expect_equal(widened.size(), nodes_k);
        angular_metric_t metric;
        widened_index_t::traversal_context_t context {stream_key(seed, label), counting_allocator_t {ledger}};
        nk::vector<util::uuid128_t> keys = make_vector<util::uuid128_t>(wanted_k);
        nk::vector<f32_t> distances = make_vector<f32_t>(wanted_k);
        for (std::size_t query = 0; query != nodes_k; ++query) {
            expect_equal(widened.search(values[query], {nodes_k}, metric, context).status, status_code_t::success_k);
            expect_true(context.results.size() >= wanted_k);
            if (context.results.size() < wanted_k) continue;
            for (std::size_t rank = 0; rank != wanted_k; ++rank) {
                auto const [widened_key, key_status] = widened.key_of(context.results[rank].node_id);
                if (expect_true(succeeded(key_status))) keys[rank] = widened_key;
                distances[rank] = context.results[rank].distance;
            }
            ranked_alike(query, to_span(keys), to_span(distances));
        }
    }

    // The view widens the same keys on storage: each counts once, and the walks rank alike.
    fu_topology_t topology = nullptr;
    expect_true(fu_topology_new(&topology) == fu_success_k);
    {
        io_queue_config_t config;
        config.queue_depth = 64;
        auto [executor, spawn_status] = widened_view_t::executor_t::spawn(topology, 1, config);
        expect_equal(spawn_status, status_code_t::success_k);
        widened_view_t viewed {forkunion_allocator_t {}};
        expect_equal(viewed.view(path, {}, embeddings_fixture_t::metric_k, executor), status_code_t::success_k);
        expect_equal(viewed.size(), nodes_k);
        nk::vector<std::size_t> entries = make_vector<std::size_t>(expected_keys.size());
        expect_equal(viewed.count(to_span(expected_keys), to_writable_span(entries), executor),
                     status_code_t::success_k);
        for (std::size_t const counted : entries) expect_equal(counted, 1u);
        auto make_metric = [](auto const&, std::size_t) noexcept {
            return expected<angular_metric_t> {angular_metric_t {}, status_code_t::success_k};
        };
        auto query_of = [&](std::size_t query, auto const&) noexcept {
            return expected<embedding_t> {values[query], status_code_t::success_k};
        };
        widened_view_t::search_config_t search_config;
        search_config.expansion_factor = nodes_k;
        nk::matrix<util::uuid128_t> keys = make_matrix<util::uuid128_t>(nodes_k, wanted_k);
        nk::matrix<f32_t> distances = make_matrix<f32_t>(nodes_k, wanted_k);
        nk::vector<std::size_t> counts = make_vector<std::size_t>(nodes_k);
        auto const [found, search_status] = viewed.search(nodes_k, make_metric, query_of, to_writable_span(keys),
                                                          to_writable_span(distances), to_writable_span(counts),
                                                          search_config, executor);
        expect_equal(search_status, status_code_t::success_k);
        for (std::size_t query = 0; query != nodes_k; ++query)
            ranked_alike(query, to_span(keys.view().row(query).as_vector()),
                         to_span(distances.view().row(query).as_vector()));
    }
    fu_topology_delete(topology);
    expect_errors("widened distances", comparison_family_t::approximate_k, comparison_verdict_t::assert_k,
                  distance_errors, 0);
    expect_equal(ledger.live_bytes, 0);
}

/** A monolith of 64-bit node IDs reaches no reader of 32-bit ones: the UUID-keyed RAM load and the
 *  UUID-keyed view both refuse it as a mismatched configuration. */
static void test_wide_nodes_refused(u64_t seed) {
    using fixture_t = embeddings_fixture<u64_t, u64_t>;
    using source_t = fixture_t::index_t;
    constexpr std::size_t nodes_k = 16;
    allocation_ledger_t ledger;
    scratch_directory_t const scratch;
    std::string const path = scratch.path("wide-nodes.safegraphs");
    serial_executor_t serial;
    {
        std::unique_ptr<source_t> const source_ptr = fixture_t::make_index(ledger);
        source_t& source = *source_ptr;
        angular_metric_t metric;
        source_t::traversal_context_t context {stream_key(seed, "wide nodes"), counting_allocator_t {ledger}};
        std::vector<embedding_t> values = fixture_t::make_values(nodes_k, stream_key(seed, "wide nodes"),
                                                                 counting_allocator_t {ledger});
        std::vector<u64_t> assigned_ids;
        expect_true(bool(source.reserve(nodes_k)));
        basic_build<fixture_t>(source, metric, values, assigned_ids, context, keep_diverse_t {4});
        expect_equal(safegraphs_write_index(std::string_view {path}, source, embedding_codec_t {}, serial,
                                            save_config_t {}, embeddings_fixture_t::metric_k),
                     status_code_t::success_k);
    }
    {
        auto [mapping, map_status] = mapped_file_t::read(path);
        expect_true(succeeded(map_status));
        widened_index_t widened {counting_allocator_t {ledger}};
        expect_equal(safegraphs_read_index(mapping.view(), widened, embedding_codec_t {}, serial, load_config_t {}),
                     status_code_t::config_mismatch_k);
    }
    fu_topology_t topology = nullptr;
    expect_true(fu_topology_new(&topology) == fu_success_k);
    {
        io_queue_config_t config;
        config.queue_depth = 64;
        auto [executor, spawn_status] = widened_view_t::executor_t::spawn(topology, 1, config);
        expect_equal(spawn_status, status_code_t::success_k);
        widened_view_t viewed {forkunion_allocator_t {}};
        expect_equal(viewed.view(path, {}, embeddings_fixture_t::metric_k, executor), status_code_t::config_mismatch_k);
        // The little-endian header length follows the magic; 2^63 bytes is past any file.
        unsigned char const header_length[8] = {0, 0, 0, 0, 0, 0, 0, 0x80};
        std::FILE* const corrupted = std::fopen(path.c_str(), "r+b");
        expect_true(corrupted != nullptr);
        expect_true(std::fseek(corrupted, 8, SEEK_SET) == 0 && std::fwrite(header_length, 1, 8, corrupted) == 8);
        std::fclose(corrupted);
        widened_view_t corrupted_view {forkunion_allocator_t {}};
        expect_equal(corrupted_view.view(path, {}, embeddings_fixture_t::metric_k, executor),
                     status_code_t::malformed_k);
    }
    fu_topology_delete(topology);
    expect_equal(ledger.live_bytes, 0);
}

/**
 *  @brief A checkpoint read as if a device held it crosses into host memory only through the
 *      caller's copy, and only as its manifest and headers, while every part keeps viewing the
 *      source; a column then copies through the same callback.
 *
 *  The copy records what it read, from the source's start: a monolith reads nothing past its
 *  data offset, and a two-part buffer nothing but its manifest and each part's header.
 */
static void test_view_reads_headers_alone(u64_t seed) {
    using index_t = embeddings_fixture_t::index_t;
    constexpr std::size_t nodes_k = 24, parts_k = 2;
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
        expect_true(bool(index->reserve(nodes_k)));
        basic_build<embeddings_fixture_t>(*index, metric, values, assigned_ids, context, keep_diverse_t {4});
        auto const [bytes, measure_status] = safegraphs_storage_requirements(
            *index, embedding_codec_t {}, save_config_t {}, embeddings_fixture_t::metric_k);
        expect_true(succeeded(measure_status));
        monolith.resize(bytes);
        expect_equal(safegraphs_write_index(bytes_span_t {monolith}, *index, embedding_codec_t {}, serial,
                                            save_config_t {}, embeddings_fixture_t::metric_k),
                     status_code_t::success_k);
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
        expect_equal(source.read_header(parts, extra_columns, device_copy, counting_allocator_t {ledger}),
                     status_code_t::success_k);
        expect_equal(parts.size(), 1u);
        if (parts.size() != 1) return;
        safegraphs_header_t const& header = parts[0].header;
        expect_equal(parts[0].nodes, nodes_k);
        expect_true(parts[0].monolith.data() == monolith.data());
        expect_true(parts[0].metric.view() == embeddings_fixture_t::metric_k);
        for (iota_view<std::size_t> const range : crossed) expect_true(range.bound <= header.data_offset);

        crossed.clear();
        std::vector<byte_t> keys(nodes_k * header.key.row_bytes()), host_keys(keys.size());
        expect_equal(source.copy_column(header, header.key, {0, nodes_k}, bytes_span_t {keys}, device_copy),
                     status_code_t::success_k);
        expect_equal(crossed.size(), 1u);
        // Host bytes copy without the callback, which this one would refuse.
        checkpoint_source_t const in_place {bytes_view_t {monolith}};
        expect_equal(
            in_place.copy_column(header, header.key, {0, nodes_k}, bytes_span_t {host_keys}, no_device_copy_t {}),
            status_code_t::success_k);
        expect_true(keys == host_keys);
    }

    // Two copies of it behind a manifest leading one buffer: the manifest and each header cross.
    std::array<std::size_t, parts_k> offsets {};
    auto part_at = [&](std::size_t part) noexcept {
        return safegraphs_part_t {{}, monolith.size(), nodes_k, offsets[part]};
    };
    auto const [buffer_bytes, layout_status] = safegraphs_buffer_layout(
        parts_k, part_at, safegraphs_alignment_default_k, std::span<std::size_t> {offsets});
    expect_equal(layout_status, status_code_t::success_k);
    std::vector<byte_t> buffer(buffer_bytes);
    expect_equal(safegraphs_write_manifest(bytes_span_t {buffer}, parts_k, part_at), status_code_t::success_k);
    for (std::size_t const offset : offsets) std::copy(monolith.begin(), monolith.end(), buffer.begin() + offset);
    origin = buffer.data();
    crossed.clear();
    {
        checkpoint_source_t const source {bytes_view_t {buffer}, 0};
        parts_t parts {counting_allocator_t {ledger}};
        extra_columns_t extra_columns {counting_allocator_t {ledger}};
        expect_equal(source.read_header(parts, extra_columns, device_copy, counting_allocator_t {ledger}),
                     status_code_t::success_k);
        expect_equal(parts.size(), parts_k);
        if (parts.size() != parts_k) return;
        for (std::size_t part = 0; part != parts_k; ++part)
            expect_true(parts[part].monolith.data() == buffer.data() + offsets[part]);
        for (iota_view<std::size_t> const range : crossed) {
            bool inside = range.bound <= offsets[0];
            for (std::size_t part = 0; part != parts_k; ++part)
                inside |= range.first >= offsets[part] && range.bound <= offsets[part] + parts[part].header.data_offset;
            expect_true(inside);
        }
        auto const [metadata, metadata_status] = safegraphs_read_metadata(source, device_copy,
                                                                          counting_allocator_t {ledger});
        expect_equal(metadata_status, status_code_t::success_k);
        expect_equal(metadata.parts, parts_k);
        expect_equal(metadata.nodes, parts_k * nodes_k);
    }
    expect_equal(ledger.live_bytes, 0);
}

/** A manifest replaces the one before it whole, through a sibling that never stays behind, while
 *  an address inside a device's span and a directory are refused, having no room for a sibling. */
static void test_manifest_replaced_durably() {
    scratch_directory_t const scratch;
    std::string const path = scratch.path("manifest.json"), staged = scratch.path("manifest.json.tmp");
    constexpr std::string_view names[] = {"part-0.safegraphs", "part-1.safegraphs", "part-2.safegraphs"};
    auto part_at = [&](std::size_t part) noexcept { return safegraphs_part_t {names[part], 4096, 8, 0}; };
    for (std::size_t const parts : {std::size_t {3}, std::size_t {2}}) {
        expect_equal(safegraphs_write_manifest(std::string_view {path}, parts, part_at), status_code_t::success_k);
        auto [mapping, map_status] = mapped_file_t::read(path);
        expect_equal(map_status, status_code_t::success_k);
        std::string_view const json {reinterpret_cast<char const*>(mapping.view().data()), mapping.size()};
        expect_true(json.find(names[parts - 1]) != std::string_view::npos);
        expect_true((json.find(names[2]) != std::string_view::npos) == (parts == 3));
        expect_true(failed(seekable_file_t::open(staged, file_access_t::read_only_k).status));
    }
    expect_equal(safegraphs_write_manifest(fmt::format("{}@4096", path), 2, part_at),
                 status_code_t::invalid_argument_k);
    expect_equal(safegraphs_write_manifest(std::string_view {"/tmp"}, 2, part_at), status_code_t::invalid_argument_k);
}

/** Rows read through a column source match the mapped bytes wherever they start, into a
 *  destination that shares their place within a page, whose whole pages come off the drive where
 *  @c TMPDIR's filesystem takes unbuffered reads, and into one that does not, which copies. */
static void test_column_rows_read_direct(u64_t seed) {
    constexpr std::size_t row_bytes = 48, rows = 1000, column_offset = 4096 + 160;
    scratch_directory_t const scratch;
    std::string const path = scratch.path("column.bin");
    {
        auto [written, write_status] = mapped_file_t::write(path, column_offset + rows * row_bytes + 77);
        expect_true(succeeded(write_status));
        if (failed(write_status)) return;
        bytes_span_t const bytes = written.span();
        for (std::size_t byte = 0; byte != bytes.size(); ++byte)
            bytes[byte] = static_cast<byte_t>((byte * 131 + seed) & 0xFF);
        expect_true(succeeded(written.sync()));
    }
    auto [mapping, map_status] = mapped_file_t::read(path);
    expect_true(succeeded(map_status));
    seekable_file_t const direct = mapping.open_direct();
    bytes_view_t const column = mapping.view().subspan(column_offset, rows * row_bytes);
    column_source_t const source {
        column, direct ? file_view_t {&direct, column_offset, column_offset + column.size()} : file_view_t {}};
    std::size_t const page = fu::ram_page_size();
    std::vector<byte_t> staged(rows * row_bytes + 3 * page);
    struct run_t {
        std::size_t first, count;
    };
    for (run_t const run : {run_t {0, rows}, run_t {256, 512}, run_t {3, 997}, run_t {999, 1}})
        for (std::size_t const shift : {std::size_t {0}, std::size_t {8}}) {
            // The destination shares the run's place within a page, or misses it by a shift.
            std::size_t const offset_in_page = (column_offset + run.first * row_bytes) % page;
            std::size_t const lead = (page - reinterpret_cast<std::uintptr_t>(staged.data()) % page) % page;
            bytes_span_t const destination = bytes_span_t {staged}.subspan(lead + offset_in_page + shift,
                                                                           run.count * row_bytes);
            expect_equal(copy_entries(source, run.first, run.count, row_bytes, destination), status_code_t::success_k);
            expect_true(std::memcmp(destination.data(), column.data() + run.first * row_bytes, destination.size()) ==
                        0);
        }
    expect_equal(copy_entries(source, 999, 2, row_bytes, bytes_span_t {staged}.first(2 * row_bytes)),
                 status_code_t::config_mismatch_k);
}

/** The shared basic-tier scenarios over the embeddings fixture - a fixed-width value column, which
 *  no modality unit has: monolith, hostile header battery, and the distributed form, each at every
 *  alignment and executor the scenarios cover - then the narrower key columns widening into UUIDs,
 *  the wider node IDs refused, and a device's checkpoint read through its headers alone. */
void test::utilities::run_safegraphs_format_cases() {
    u64_t const seed = test_run.settings.seed;
    run_test_case("test_basic_safegraphs_round_trip<embeddings_fixture_t>",
                  test_basic_safegraphs_round_trip<embeddings_fixture_t>, seed, "u64", embeddings_fixture_t::metric_k);
    run_test_case("test_basic_safegraphs_hostile<embeddings_fixture_t>",
                  test_basic_safegraphs_hostile<embeddings_fixture_t>, seed, embeddings_fixture_t::metric_k);
    run_test_case("test_basic_safegraphs_distributed<embeddings_fixture_t>",
                  test_basic_safegraphs_distributed<embeddings_fixture_t>, seed, embeddings_fixture_t::metric_k);
    run_test_case("test_keys_widen<embeddings_fixture_t>", test_keys_widen<embeddings_fixture_t>, seed, "u64 keys");
    run_test_case("test_keys_widen<embeddings_fixture<u32_t, u32_t>>",
                  test_keys_widen<embeddings_fixture<u32_t, u32_t>>, seed, "u32 keys");
    run_test_case("test_wide_nodes_refused", test_wide_nodes_refused, seed);
    run_test_case("test_view_reads_headers_alone", test_view_reads_headers_alone, seed);
    run_test_case("test_manifest_replaced_durably", test_manifest_replaced_durably);
    run_test_case("test_column_rows_read_direct", test_column_rows_read_direct, seed);
}

#pragma endregion Embeddings
