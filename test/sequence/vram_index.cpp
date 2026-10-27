/**
 *  @file test/sequence/vram_index.cpp
 *  @author Ash Vardanian
 *  @date October 15, 2026
 *  @brief The sequence VRAM cell: byte and rune Levenshtein and window overlap on an NVIDIA GPU
 *      beside the RAM index on the same data, judged by recall against exact neighbors, every
 *      reported distance against the serial kit the RAM suite checks against its oracle.
 *
 *  Strings are planted templates over the nucleotides with an eighth of their symbols edited, so
 *  neighbors cluster by template while the bytes differ. One collection keeps to the two-word kit,
 *  another needs the four-word one, and the last needs a block per walk, its rows past 1 KiB split
 *  over four lanes each, so the walks, the selections and the merges run three kits. A fourth
 *  spells its nucleotides as runes of one to three bytes, for the rune kits, and a fifth scores
 *  window overlap at two widths, a warp to a row. The last two align nucleotides packed two bits
 *  apiece, one query a stripe and the longer ones several. Rows insert in three batches whose
 *  tapes start past their own first byte, as an Arrow slice's do, so the later batches walk graphs
 *  whose reverse links earlier batches merged.
 */
#include <algorithm> // `std::min`, `std::nth_element`
#include <array>     // `std::array` - the templates
#include <limits>    // `std::numeric_limits` - the node-ID type's sentinel
#include <memory>    // `std::unique_ptr` - a plural's shards
#include <new>       // `std::nothrow` - a plural's shards
#include <string>    // `std::string` - the tape and the generated strings
#include <vector>    // `std::vector` - strings, offsets, matches

#include "usearch/sequence_index.h"
#include "usearch/sequence/ram_index.hpp"
#if USEARCH_ARCH_CUDA_
#include "usearch/sequence/vram_nvidia_index.hpp"
#include "usearch/basic/vram_nvidia_indexes.hpp"
#endif

#include "harness.hpp"

#if USEARCH_ARCH_CUDA_

/** The library's fatbin of one kind's sequence kits, defined in
 *  @c src/sequence/vram_variants_nvidia.cpp. */
template <usearch_sequence_metric_t metric_>
struct sequence_nvidia_fatbin {
    static us::utilities::bytes_view_t bytes() noexcept;
};

#endif

namespace unum::usearch::test::sequence {
using namespace unum::usearch::sequence;

#if USEARCH_ARCH_CUDA_

/** Draws @p count strings of about @p length nucleotides from eight templates, an eighth of every
 *  string's symbols substituted, inserted or erased. */
static std::vector<std::string> generate_nucleotides(std::size_t count, std::size_t length, u64_t seed) {
    constexpr std::string_view nucleotides_k = "ACGT";
    constexpr std::size_t templates_k = 8;
    splitmix64_t random_generator {seed};
    uniform_below<u32_t> const symbol_distribution {static_cast<u32_t>(nucleotides_k.size())};
    uniform_below<u32_t> const template_distribution {templates_k};
    uniform_below<u32_t> const edit_kind_distribution {3};
    std::array<std::string, templates_k> templates;
    for (std::string& symbols : templates)
        for (std::size_t symbol = 0; symbol != length; ++symbol)
            symbols += nucleotides_k[symbol_distribution(random_generator)];
    std::vector<std::string> strings(count);
    for (std::string& symbols : strings) {
        symbols = templates[template_distribution(random_generator)];
        for (std::size_t edit = 0; edit != length / 8; ++edit) {
            std::size_t const position = uniform_below<u32_t> {static_cast<u32_t>(symbols.size())}(random_generator);
            char const symbol = nucleotides_k[symbol_distribution(random_generator)];
            switch (edit_kind_distribution(random_generator)) {
            case 0: symbols[position] = symbol; break;
            case 1: symbols.insert(symbols.begin() + position, symbol); break;
            default: symbols.erase(symbols.begin() + position); break;
            }
        }
    }
    return strings;
}

/** Respells every nucleotide as a rune of one to three bytes, so edits count runes, not bytes. */
static void respell_as_runes(std::vector<std::string>& strings) {
    for (std::string& symbols : strings) {
        std::string runes;
        for (char const symbol : symbols) switch (symbol) {
            case 'A': runes += "A"; break;
            case 'C': runes += "\xC3\x87"; break;
            case 'G': runes += "\xC4\x9E"; break;
            default: runes += "\xE0\xB8\x97"; break;
            }
        symbols = std::move(runes);
    }
}

/** Packs every string's nucleotides two bits a code, A, C, G and T in that order. */
static void pack_as_codes(std::vector<std::string>& strings) {
    constexpr std::string_view nucleotides_k = "ACGT";
    for (std::string& symbols : strings) {
        std::vector<u8_t> codes;
        for (char const symbol : symbols) codes.push_back(static_cast<u8_t>(nucleotides_k.find(symbol)));
        std::string row(packed_codes_bytes(codes.size(), 2), '\0');
        pack_codes(codes, 2, bytes_of(row.data(), row.size()));
        symbols = std::move(row);
    }
}

/** The window widths the overlap cell hashes at, in bytes. */
static constexpr std::size_t vram_overlap_widths_k[] = {4, 8};

/** The Needleman-Wunsch cell's costs over the nucleotides: transitions one, transversions and gaps
 *  two. */
static constexpr needleman_wunsch_costs_t vram_nucleotide_costs() noexcept {
    needleman_wunsch_costs_t costs {};
    costs.code_bits = 2, costs.classes = 4, costs.gap = 2;
    for (std::size_t first = 0; first != 4; ++first)
        for (std::size_t second = 0; second != 4; ++second)
            costs.substitutions[first][second] = first == second ? 0 : (first & 1) == (second & 1) ? 1 : 2;
    return costs;
}

static constexpr needleman_wunsch_costs_t vram_nucleotide_costs_k = vram_nucleotide_costs();

/** The metric @p metric_ scores with on the CPU under @p capability_, sized for @p staging
 *  candidates. */
template <usearch_sequence_metric_t metric_, sz_capability_t capability_>
static auto make_sequence_metric(std::size_t staging, std::size_t& failures) {
    if constexpr (metric_ == usearch_sequence_metric_overlap_k) {
        overlap_metric<typename overlap_ladder<capability_>::kit_t> metric;
        failures += expect_eq(metric.init(staging, vram_overlap_widths_k), status_t::success_k);
        return metric;
    }
    else if constexpr (metric_ == usearch_sequence_metric_needleman_wunsch_k) {
        needleman_wunsch_metric<typename needleman_wunsch_ladder<capability_>::kit_t> metric;
        failures += expect_eq(metric.init(staging, vram_nucleotide_costs_k), status_t::success_k);
        return metric;
    }
    else {
        constexpr sz_levenshtein_symbol_t symbol_k = metric_ == usearch_sequence_metric_levenshtein_utf8_k
                                                         ? sz_levenshtein_runes_k
                                                         : sz_levenshtein_bytes_k;
        levenshtein_metric<typename levenshtein_ladder<symbol_k, capability_>::kit_t> metric;
        failures += expect_eq(metric.init(staging), status_t::success_k);
        return metric;
    }
}

/** Matches, of @p wanted at most, whose distances reach no further than the @p wanted -th exact
 *  one: ties make the IDs ambiguous, the distances never. */
static std::size_t matched_count(std::span<sz_size_t const> found, std::vector<sz_size_t> exact, std::size_t wanted) {
    std::nth_element(exact.begin(), exact.begin() + (wanted - 1), exact.end());
    std::size_t matched = 0;
    for (sz_size_t const distance : found) matched += distance <= exact[wanted - 1];
    return min_of(matched, wanted);
}

/** What a GPU collection's recall is held to: a RAM index built at the same breadth, or nothing
 *  beyond the exact distances, where diversity pruning on the host would take minutes. */
enum class recall_reference_t {
    ram_index_k,
    exact_distances_k,
};

/**
 *  @brief One collection of @p count strings of about @p length bytes, held out @p queries_count
 *      queries, built on the GPU with @p node_id_type_ IDs and in RAM alike.
 *
 *  Every match the GPU reports must carry the pair's exact distance, ascending, and under
 *  @p reference the GPU's recall at ten must reach the RAM index's at the same breadth, give or
 *  take a twentieth.
 */
template <usearch_sequence_metric_t metric_, typename node_id_type_ = u32_t>
static std::size_t check_sequence_vram_collection(u64_t seed, cuda_executor_t& executor, std::size_t count,
                                                  std::size_t length, std::size_t queries_count,
                                                  recall_reference_t reference = recall_reference_t::ram_index_k) {
    using ram_index_t = ram_index<us::utilities::uuid_t, u32_t>;
    using vram_index_t = vram_nvidia_index<metric_, node_id_type_, sz_cap_cuda_k, sequence_nvidia_fatbin<metric_>>;
    constexpr std::size_t wanted_k = 10, expansion_k = 64, batches_k = 3, origin_k = 3;
    std::size_t failures = 0;

    std::vector<std::string> strings = generate_nucleotides(count + queries_count, length, seed);
    if constexpr (metric_ == usearch_sequence_metric_levenshtein_utf8_k) respell_as_runes(strings);
    if constexpr (metric_ == usearch_sequence_metric_needleman_wunsch_k) pack_as_codes(strings);
    std::string tape(origin_k, '\0');
    std::vector<std::size_t> offsets {origin_k};
    for (std::string const& symbols : strings) tape += symbols, offsets.push_back(tape.size());
    byte_t const* const tape_bytes = reinterpret_cast<byte_t const*>(tape.data());

    // The exact distances of every query to every stored string, through the serial kit.
    std::vector<std::vector<sz_size_t>> exact(queries_count, std::vector<sz_size_t>(count));
    auto serial_metric = make_sequence_metric<metric_, sz_cap_serial_k>(count, failures);
    for (std::size_t query = 0; query != queries_count; ++query) {
        serial_metric.reset(std::string_view {strings[count + query]});
        for (std::size_t row = 0; row != count; ++row) serial_metric.push(std::string_view {strings[row]});
        serial_metric.drain(exact[query], us::basic::unbounded_radius<sz_size_t>());
    }

    // The RAM index over the same rows, at the same breadth, which an exact-distance reference
    // leaves unbuilt and its recall at zero.
    std::size_t ram_matched = 0;
    if (reference == recall_reference_t::ram_index_k) {
        ram_index_t ram_index;
        auto metric = make_sequence_metric<metric_, sz_cap_any_k>(expansion_k * connectivity_32_k, failures);
        ram_index_t::traversal_context_t ram_context {seed};
        failures += expect(bool(ram_index.reserve(count)));
        for (std::size_t row = 0; row != count; ++row) {
            ram_index_t::value_t value;
            [[maybe_unused]] sz::status_t const assigned = value.assign(strings[row]);
            us::utilities::uuid_t const key = uuid_of(row);
            failures += expect_eq(ram_index
                                      .insert({&value, 1}, {&key, 1}, metric, {}, ram_context,
                                              us::basic::keep_diverse_t {connectivity_32_k}, {expansion_k})
                                      .status,
                                  status_t::success_k);
        }
        for (std::size_t query = 0; query != queries_count; ++query) {
            failures += expect_eq(
                ram_index.search(std::string_view {strings[count + query]}, metric, ram_context, {expansion_k}).status,
                status_t::success_k);
            std::vector<sz_size_t> found;
            for (std::size_t rank = 0; rank != min_of(wanted_k, ram_context.results.size()); ++rank)
                found.push_back(ram_context.results[rank].distance);
            ram_matched += matched_count(found, exact[query], wanted_k);
        }
    }

    // The GPU index, in three batches, each over a context carved from the executor's run.
    int const device = executor.device(0);
    vram_index_t vram_index {cuda_virtual_memory_allocator<byte_t> {device}};
    if constexpr (metric_ == usearch_sequence_metric_overlap_k)
        failures += expect_eq(vram_index.window_widths(vram_overlap_widths_k), status_t::success_k);
    if constexpr (metric_ == usearch_sequence_metric_needleman_wunsch_k)
        failures += expect_eq(vram_index.costs(vram_nucleotide_costs_k), status_t::success_k);
    failures += expect_eq(vram_index.reserve({count, connectivity_32_k, offsets[count] - offsets[0]}).status,
                          status_t::success_k);
    typename vram_index_t::insert_config_t const insert_config {expansion_k};
    for (std::size_t batch = 0; batch != batches_k && !failures; ++batch) {
        std::size_t const first = count * batch / batches_k, bound = count * (batch + 1) / batches_k;
        std::span<std::size_t const> const batch_offsets {offsets.data() + first, bound - first + 1};
        failures += expect_eq(
            executor.reserve(vram_index.traversal_memory_usage({tape_bytes, batch_offsets}, insert_config)),
            status_t::success_k);
        typename vram_index_t::traversal_context_t context {
            cuda_arena_allocator_t {executor.stream(0).arena, executor.stream(0)}};
        auto const [inserted, status] = vram_index.insert({tape_bytes, batch_offsets}, context, insert_config);
        failures += expect_eq(status, status_t::success_k);
        failures += expect_eq(inserted.first_id, first);
        failures += expect_eq(inserted.new_size, bound);
        failures += expect(inserted.computed_distances > 0);
    }
    failures += expect_eq(vram_index.size(), count);

    std::vector<node_id_type_> ids(queries_count * wanted_k);
    std::vector<f32_t> distances(queries_count * wanted_k);
    std::span<std::size_t const> const query_offsets {offsets.data() + count, queries_count + 1};
    typename vram_index_t::search_config_t const search_config {expansion_k, wanted_k};
    failures += expect_eq(
        executor.reserve(vram_index.traversal_memory_usage({tape_bytes, query_offsets}, search_config)),
        status_t::success_k);
    typename vram_index_t::traversal_context_t context {
        cuda_arena_allocator_t {executor.stream(0).arena, executor.stream(0)}};
    if (!failures)
        failures += expect_eq(vram_index
                                  .search({tape_bytes, query_offsets}, {ids.data(), {queries_count, wanted_k}},
                                          {distances.data(), {queries_count, wanted_k}}, context, search_config)
                                  .status,
                              status_t::success_k);
    std::size_t vram_matched = 0;
    for (std::size_t query = 0; query != queries_count && !failures; ++query) {
        std::vector<sz_size_t> found;
        for (std::size_t rank = 0; rank != wanted_k; ++rank) {
            node_id_type_ const id = ids[query * wanted_k + rank];
            if (id == std::numeric_limits<node_id_type_>::max()) break;
            // The candidates keep 31 bits of a float, so the overlap kind's fixed point loses a
            // mantissa bit more on the device.
            f32_t const distance = distances[query * wanted_k + rank];
            if (expect(id < count)) {
                ++failures;
                break;
            }
            failures += expect_near(distance, static_cast<f32_t>(exact[query][id]));
            failures += expect(found.empty() || found.back() <= exact[query][id]);
            found.push_back(exact[query][id]);
        }
        failures += expect_eq(found.size(), wanted_k);
        vram_matched += matched_count(found, exact[query], wanted_k);
    }
    f64_t const ram_recall = static_cast<f64_t>(ram_matched) / (queries_count * wanted_k);
    f64_t const vram_recall = static_cast<f64_t>(vram_matched) / (queries_count * wanted_k);
    if (vram_recall + 0.05 < ram_recall)
        fmt::print(stderr, "  recall at ten on {} strings of {} bytes: {:.3f} on the GPU, {:.3f} in RAM\n", count,
                   length, vram_recall, ram_recall);
    failures += expect(vram_recall + 0.05 >= ram_recall);
    return failures;
}

/** Builds the byte Levenshtein shards of a plural, unreserved, on the device it names. */
struct sequence_vram_shards_t {
    using index_t = vram_nvidia_index<usearch_sequence_metric_levenshtein_bytes_k, u32_t, sz_cap_cuda_k,
                                      sequence_nvidia_fatbin<usearch_sequence_metric_levenshtein_bytes_k>>;

    std::unique_ptr<index_t> operator()(int device) const noexcept {
        return std::unique_ptr<index_t>(new (std::nothrow) index_t {cuda_virtual_memory_allocator<byte_t> {device}});
    }
};

/** Two shards on one device bounded at 1000 rows take 5000 strings in one batch, which opens two
 *  more generations of two: every key counts once across the six shards, and the merged recall at
 *  ten reaches one unbounded shard's, give or take a twentieth. */
static std::size_t check_sequence_vram_plural(u64_t seed, cuda_executor_t& executor) {
    using index_t = sequence_vram_shards_t::index_t;
    using indexes_t = us::basic::vram_nvidia_indexes<index_t, us::utilities::uuid_t, sequence_vram_shards_t>;
    constexpr std::size_t rows_k = 5000, shard_capacity_k = 1000, queries_k = 64, wanted_k = 10, expansion_k = 64;
    std::size_t failures = 0;

    std::vector<std::string> const strings = generate_nucleotides(rows_k + queries_k, 100, seed);
    std::string tape;
    std::vector<std::size_t> offsets {0};
    for (std::string const& symbols : strings) tape += symbols, offsets.push_back(tape.size());
    byte_t const* const tape_bytes = reinterpret_cast<byte_t const*>(tape.data());
    tape_view<byte_t const> const rows {tape_bytes, {offsets.data(), rows_k + 1}};
    tape_view<byte_t const> const queries {tape_bytes, {offsets.data() + rows_k, queries_k + 1}};
    std::vector<std::vector<sz_size_t>> exact(queries_k, std::vector<sz_size_t>(rows_k));
    auto serial_metric = make_sequence_metric<usearch_sequence_metric_levenshtein_bytes_k, sz_cap_serial_k>(rows_k,
                                                                                                            failures);
    for (std::size_t query = 0; query != queries_k; ++query) {
        serial_metric.reset(std::string_view {strings[rows_k + query]});
        for (std::size_t row = 0; row != rows_k; ++row) serial_metric.push(std::string_view {strings[row]});
        serial_metric.drain(exact[query], us::basic::unbounded_radius<sz_size_t>());
    }
    std::vector<us::utilities::uuid_t> keys(rows_k);
    for (std::size_t row = 0; row != rows_k; ++row) keys[row] = uuid_of(row);

    indexes_t indexes {forkunion_allocator_t {no_memory_domain_k}, sequence_vram_shards_t {}};
    indexes.bound_shards(shard_capacity_k);
    failures += expect_eq(indexes.emplace(executor.stream(0).device), status_t::success_k);
    failures += expect_eq(indexes.emplace(executor.stream(0).device), status_t::success_k);
    failures += expect_eq(indexes.reserve({rows_k, connectivity_32_k, offsets[rows_k]}).status, status_t::success_k);
    auto const [inserted, insert_status] = indexes.insert(keys, rows, executor, {expansion_k});
    failures += expect_eq(insert_status, status_t::success_k);
    failures += expect(inserted.computed_distances > 0);
    failures += expect_eq(indexes.shards_count(), std::size_t {6});
    failures += expect_eq(indexes.size(), rows_k);
    std::size_t miscounted = 0;
    for (us::utilities::uuid_t const& key : keys) miscounted += indexes.count(key) != 1;
    failures += expect_eq(miscounted, std::size_t {0});

    std::vector<us::utilities::uuid_t> found_keys(queries_k * wanted_k);
    std::vector<f32_t> found_distances(queries_k * wanted_k);
    std::vector<std::size_t> counts(queries_k);
    typename index_t::search_config_t const search_config {expansion_k, wanted_k};
    failures += expect_eq(indexes
                              .search(queries, executor, search_config, {found_keys.data(), {queries_k, wanted_k}},
                                      nk::matrix_span<f32_t> {found_distances.data(), {queries_k, wanted_k}}, counts)
                              .status,
                          status_t::success_k);
    std::size_t plural_matched = 0;
    for (std::size_t query = 0; query != queries_k; ++query) {
        std::vector<sz_size_t> found;
        for (std::size_t rank = 0; rank != counts[query]; ++rank)
            found.push_back(static_cast<sz_size_t>(found_distances[query * wanted_k + rank]));
        plural_matched += matched_count(found, exact[query], wanted_k);
    }

    index_t single {cuda_virtual_memory_allocator<byte_t> {executor.device(0)}};
    failures += expect_eq(single.reserve({rows_k, connectivity_32_k, offsets[rows_k]}).status, status_t::success_k);
    typename index_t::insert_config_t const insert_config {expansion_k};
    failures += expect_eq(executor.reserve(max_of(single.traversal_memory_usage(rows, insert_config),
                                                  single.traversal_memory_usage(queries, search_config))),
                          status_t::success_k);
    typename index_t::traversal_context_t context {
        cuda_arena_allocator_t {executor.stream(0).arena, executor.stream(0)}};
    failures += expect_eq(single.insert(rows, context, insert_config).status, status_t::success_k);
    std::vector<u32_t> ids(queries_k * wanted_k);
    failures += expect_eq(single
                              .search(queries, {ids.data(), {queries_k, wanted_k}},
                                      {found_distances.data(), {queries_k, wanted_k}}, context, search_config)
                              .status,
                          status_t::success_k);
    std::size_t single_matched = 0;
    for (std::size_t query = 0; query != queries_k; ++query) {
        std::vector<sz_size_t> found;
        for (std::size_t rank = 0;
             rank != wanted_k && ids[query * wanted_k + rank] != std::numeric_limits<u32_t>::max(); ++rank)
            found.push_back(static_cast<sz_size_t>(found_distances[query * wanted_k + rank]));
        single_matched += matched_count(found, exact[query], wanted_k);
    }
    failures += expect(plural_matched * 20 + queries_k * wanted_k >= single_matched * 20);
    return failures;
}

std::size_t check_sequence_vram(settings_t const& settings, gpu_device_t const& device) {
    if (device.vendor != gpu_vendor_t::nvidia_k) return 0;
    std::size_t failures = 0;
    int const ordinal = static_cast<int>(device.vendor_ordinal);
    auto [executor, spawn_status] = cuda_executor_t::spawn(std::span<int const> {&ordinal, 1});
    if (failed(spawn_status)) return fmt::println(stderr, "- CUDA: device #{} opens no stream", ordinal), 1;
    auto const run = [&](char const* name, auto&& check) {
        if (settings.filter.selects(name)) failures += report(name, check());
    };
    constexpr usearch_sequence_metric_t bytes_k = usearch_sequence_metric_levenshtein_bytes_k;
    constexpr usearch_sequence_metric_t runes_k = usearch_sequence_metric_levenshtein_utf8_k;
    constexpr usearch_sequence_metric_t overlap_k = usearch_sequence_metric_overlap_k;
    constexpr usearch_sequence_metric_t aligned_k = usearch_sequence_metric_needleman_wunsch_k;
    run("sequence_vram 100 nt", [&] {
        return check_sequence_vram_collection<bytes_k>(stream_key(settings.seed, "short"), executor, 2048, 100, 128);
    });
    run("sequence_vram 200 nt", [&] {
        return check_sequence_vram_collection<bytes_k>(stream_key(settings.seed, "long"), executor, 512, 200, 64);
    });
    run("sequence_vram 1100 nt", [&] {
        return check_sequence_vram_collection<bytes_k>(stream_key(settings.seed, "block"), executor, 256, 1100, 32);
    });
    run("sequence_vram 200 runes", [&] {
        return check_sequence_vram_collection<runes_k>(stream_key(settings.seed, "runes"), executor, 512, 200, 64);
    });
    run("sequence_vram 200 nt overlap", [&] {
        return check_sequence_vram_collection<overlap_k>(stream_key(settings.seed, "overlap"), executor, 512, 200, 64);
    });
    run("sequence_vram 200 nt needleman-wunsch", [&] {
        return check_sequence_vram_collection<aligned_k>(stream_key(settings.seed, "aligned"), executor, 512, 200, 64);
    });
    // Strided alignments past one tile only need to match the exact costs; a host index over them
    // would spend its minutes on diversity pruning.
    run("sequence_vram 1100 nt needleman-wunsch", [&] {
        return check_sequence_vram_collection<aligned_k>(stream_key(settings.seed, "aligned_stripes"), executor, 256,
                                                         1100, 8, recall_reference_t::exact_distances_k);
    });
    run("sequence_vram plural",
        [&] { return check_sequence_vram_plural(stream_key(settings.seed, "plural"), executor); });
    return failures;
}

#else

std::size_t check_sequence_vram(settings_t const&, gpu_device_t const&) { return 0; }

#endif

} // namespace unum::usearch::test::sequence
