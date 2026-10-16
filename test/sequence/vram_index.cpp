/**
 *  @file test/sequence/vram_index.cpp
 *  @author Ash Vardanian
 *  @date October 15, 2026
 *  @brief The sequence VRAM cell: byte and rune Levenshtein on an NVIDIA GPU beside the RAM index
 *      on the same data, judged by recall against exact neighbors, every reported distance against
 *      the serial kit the RAM suite checks against its oracle.
 *
 *  Strings are planted templates over the nucleotides with an eighth of their symbols edited, so
 *  neighbors cluster by template while the bytes differ. One collection keeps to the two-word kit,
 *  another needs the four-word one, and the last needs a block per walk, its rows past 1 KiB split
 *  over four lanes each, so the walks, the selections and the merges run three kits. A fourth
 *  spells its nucleotides as runes of one to three bytes, for the rune kits. Rows insert in three
 *  batches whose tapes start past their own first byte, as an Arrow slice's do, so the later
 *  batches walk graphs whose reverse links earlier batches merged.
 */
#include <algorithm> // `std::min`, `std::nth_element`
#include <array>     // `std::array` - the templates
#include <limits>    // `std::numeric_limits` - the node-ID type's sentinel
#include <string>    // `std::string` - the tape and the generated strings
#include <vector>    // `std::vector` - strings, offsets, matches

#include "usearch/sequence/ram_index.hpp"
#if USEARCH_ARCH_CUDA_
#include "usearch/sequence/vram_nvidia_index.hpp"
#endif

#include "harness.hpp"

#if USEARCH_ARCH_CUDA_

/** The library's fatbin of every sequence kit, defined in
 *  @c src/sequence/vram_variants_nvidia.cu. */
struct sequence_nvidia_fatbin_t {
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

/** Matches, of @p wanted at most, whose distances reach no further than the @p wanted -th exact
 *  one: ties make the IDs ambiguous, the distances never. */
static std::size_t matched_count(std::span<i64_t const> found, std::vector<i64_t> exact, std::size_t wanted) {
    std::nth_element(exact.begin(), exact.begin() + (wanted - 1), exact.end());
    std::size_t matched = 0;
    for (i64_t const distance : found) matched += distance <= exact[wanted - 1];
    return min_of(matched, wanted);
}

/**
 *  @brief One collection of @p count strings of about @p length bytes, held out @p queries_count
 *      queries, built on the GPU with @p node_id_type_ IDs and in RAM alike.
 *
 *  Every match the GPU reports must carry the pair's exact distance, ascending, and the GPU's
 *  recall at ten must reach the RAM index's at the same breadth, give or take a twentieth.
 */
template <sz_levenshtein_symbol_t symbol_, typename node_id_type_ = u32_t>
static std::size_t check_sequence_vram_collection(u64_t seed, cuda_executor_t& executor, std::size_t count,
                                                  std::size_t length, std::size_t queries_count) {
    using ram_index_t = ram_index<us::utilities::uuid_t, u32_t>;
    using vram_index_t = vram_nvidia_index<symbol_, node_id_type_, sz_cap_cuda_k, sequence_nvidia_fatbin_t>;
    using metric_t = levenshtein_metric<sz_cap_any_k, symbol_>;
    using serial_metric_t = levenshtein_metric<sz_cap_serial_k, symbol_>;
    constexpr std::size_t wanted_k = 10, expansion_k = 64, batches_k = 3, origin_k = 3;
    std::size_t failures = 0;

    std::vector<std::string> strings = generate_nucleotides(count + queries_count, length, seed);
    if constexpr (symbol_ == sz_levenshtein_runes_k) respell_as_runes(strings);
    std::string tape(origin_k, '\0');
    std::vector<std::size_t> offsets {origin_k};
    for (std::string const& symbols : strings) tape += symbols, offsets.push_back(tape.size());
    byte_t const* const tape_bytes = reinterpret_cast<byte_t const*>(tape.data());

    // The exact distances of every query to every stored string, through the serial kit.
    std::vector<std::vector<i64_t>> exact(queries_count, std::vector<i64_t>(count));
    serial_metric_t serial_metric;
    failures += expect_eq(serial_metric.init(count), status_t::success_k);
    for (std::size_t query = 0; query != queries_count; ++query) {
        serial_metric.reset(std::string_view {strings[count + query]});
        for (std::size_t row = 0; row != count; ++row) serial_metric.push(std::string_view {strings[row]});
        serial_metric.drain(exact[query], us::basic::unbounded_radius<i64_t>());
    }

    // The RAM index over the same rows, at the same breadth.
    ram_index_t ram_index;
    metric_t metric;
    failures += expect_eq(metric.init(expansion_k * connectivity_32_k), status_t::success_k);
    ram_index_t::traversal_context_t ram_context {seed};
    failures += expect(bool(ram_index.reserve(count)));
    for (std::size_t row = 0; row != count; ++row) {
        ram_index_t::value_t value;
        [[maybe_unused]] sz::status_t const assigned = value.assign(strings[row]);
        us::utilities::uuid_t const key = uuid_of(row);
        failures += expect_eq(ram_index
                                  .insert({&value, 1}, {&key, 1}, metric, {}, ram_context,
                                          us::basic::keep_diverse_t {connectivity_32_k}, {expansion_k, 0})
                                  .status,
                              status_t::success_k);
    }
    std::size_t ram_matched = 0;
    for (std::size_t query = 0; query != queries_count; ++query) {
        failures += expect_eq(
            ram_index.search(std::string_view {strings[count + query]}, metric, ram_context, {expansion_k, 0}).status,
            status_t::success_k);
        std::vector<i64_t> found;
        for (std::size_t rank = 0; rank != min_of(wanted_k, ram_context.results.size()); ++rank)
            found.push_back(ram_context.results[rank].distance);
        ram_matched += matched_count(found, exact[query], wanted_k);
    }

    // The GPU index, in three batches, each over a context carved from the executor's run.
    int const device = executor.device(0);
    vram_index_t vram_index {cuda_virtual_memory_allocator<byte_t> {device}};
    failures += expect_eq(vram_index.reserve({count, connectivity_32_k, offsets[count] - offsets[0]}).status,
                          status_t::success_k);
    typename vram_index_t::insert_config_t const insert_config {expansion_k};
    for (std::size_t batch = 0; batch != batches_k && !failures; ++batch) {
        std::size_t const first = count * batch / batches_k, bound = count * (batch + 1) / batches_k;
        std::span<std::size_t const> const batch_offsets {offsets.data() + first, bound - first + 1};
        failures += expect_eq(
            executor.reserve(vram_index.traversal_memory_usage(tape_bytes, batch_offsets, insert_config)),
            status_t::success_k);
        typename vram_index_t::traversal_context_t context {
            cuda_arena_allocator_t {executor.stream(0).arena, executor.stream(0)}};
        auto const [inserted, status] = vram_index.insert(tape_bytes, batch_offsets, context, insert_config);
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
    failures += expect_eq(executor.reserve(vram_index.traversal_memory_usage(tape_bytes, query_offsets, search_config)),
                          status_t::success_k);
    typename vram_index_t::traversal_context_t context {
        cuda_arena_allocator_t {executor.stream(0).arena, executor.stream(0)}};
    if (!failures)
        failures += expect_eq(vram_index
                                  .search(tape_bytes, query_offsets, {ids.data(), {queries_count, wanted_k}},
                                          {distances.data(), {queries_count, wanted_k}}, context, search_config)
                                  .status,
                              status_t::success_k);
    std::size_t vram_matched = 0;
    for (std::size_t query = 0; query != queries_count && !failures; ++query) {
        std::vector<i64_t> found;
        for (std::size_t rank = 0; rank != wanted_k; ++rank) {
            node_id_type_ const id = ids[query * wanted_k + rank];
            if (id == std::numeric_limits<node_id_type_>::max()) break;
            i64_t const distance = static_cast<i64_t>(distances[query * wanted_k + rank]);
            failures += expect(id < count) || expect_eq(distance, exact[query][id]);
            failures += expect(found.empty() || found.back() <= distance);
            found.push_back(distance);
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

std::size_t check_sequence_vram(settings_t const& settings, gpu_device_t const& device) {
    if (device.vendor != gpu_vendor_t::nvidia_k || !settings.selects("sequence_vram")) return 0;
    int const ordinal = static_cast<int>(device.vendor_ordinal);
    auto [executor, spawn_status] = cuda_executor_t::spawn(std::span<int const> {&ordinal, 1});
    if (failed(spawn_status)) return fmt::println(stderr, "- CUDA: device #{} opens no stream", ordinal), 1;
    std::size_t failures = 0;
    failures += report("sequence_vram 100 nt", check_sequence_vram_collection<sz_levenshtein_bytes_k>(
                                                   stream_key(settings.seed, "short"), executor, 2048, 100, 128));
    failures += report("sequence_vram 200 nt", check_sequence_vram_collection<sz_levenshtein_bytes_k>(
                                                   stream_key(settings.seed, "long"), executor, 512, 200, 64));
    failures += report("sequence_vram 1100 nt", check_sequence_vram_collection<sz_levenshtein_bytes_k>(
                                                    stream_key(settings.seed, "block"), executor, 256, 1100, 32));
    failures += report("sequence_vram 200 runes", check_sequence_vram_collection<sz_levenshtein_runes_k>(
                                                      stream_key(settings.seed, "runes"), executor, 512, 200, 64));
    return failures;
}

#else

std::size_t check_sequence_vram(settings_t const&, gpu_device_t const&) { return 0; }

#endif

} // namespace unum::usearch::test::sequence
