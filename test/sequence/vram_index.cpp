/**
 *  @file test/sequence/vram_index.cpp
 *  @author Ash Vardanian
 *  @date October 15, 2026
 *  @brief The sequence VRAM cell: byte and rune Levenshtein and window overlap on an NVIDIA, AMD or
 *      Apple GPU beside the RAM index on the same data, judged by recall against exact neighbors,
 *      every reported distance against the serial kit the RAM suite checks against its oracle.
 *
 *  Sequences are planted templates over the nucleotides with an eighth of their symbols edited, so
 *  neighbors cluster by template while the bytes differ. One collection keeps to the two-word kit,
 *  another needs the four-word one, and the last needs a block per walk, its sequences past 1 KiB
 *  split over four lanes each, so the walks, the selections and the merges run three kits. A fourth
 *  spells its nucleotides as runes of one to three bytes, for the rune kits, and a fifth scores
 *  window overlap at two widths, a warp to an entry. The last two align nucleotides packed two bits
 *  apiece, one query a stripe and the longer ones several. Sequences insert in three batches whose
 *  tapes start past their own first byte, as an Arrow slice's do, so the later batches walk graphs
 *  whose reverse links earlier batches merged.
 */
#undef NDEBUG // ! A test's oracle must stay live in every build

#include <cstdio>  // `std::remove` - a reloaded checkpoint's files
#include <cstring> // `std::memcpy`, `std::memcmp` - a query's exact distances, a gathered sequence

#include <algorithm>   // `std::nth_element`
#include <array>       // `std::array` - the templates
#include <limits>      // `std::numeric_limits` - the node-ID type's sentinel
#include <memory>      // `std::unique_ptr` - a sharded index's shards
#include <new>         // `std::nothrow` - a sharded index's shards
#include <string>      // `std::string` - the tape and the generated strings
#include <string_view> // `std::string_view` - a case label's arguments
#include <type_traits> // `std::conditional_t` - the GPU kit of a kind
#include <vector>      // `std::vector` - strings, offsets, found distances

#include "usearch/sequence_index.h"
#include "usearch/sequence/ram_index.hpp"
#if USEARCH_ARCH_CUDA_
#include "usearch/sequence/vram_nvidia_metric.cuh"
#include "usearch/basic/vram_nvidia_indexes.hpp"
#endif
#if USEARCH_ARCH_ROCM_
#include "usearch/sequence/vram_amd_metric.hip"
#include "usearch/basic/vram_amd_indexes.hpp"
#include "lib.hpp" // `gpu_class_t`, `stringzilla_capabilities_of` - the class the AMD cells run
#endif
#if USEARCH_ARCH_METAL_
#include "usearch/sequence/vram_apple_index.hpp"
#endif

#include "harness.hpp"

#if USEARCH_ARCH_CUDA_

namespace util = unum::usearch::utilities;

/** The library's fatbin of one kind's sequence kits, defined in
 *  @c src/sequence/vram_variants_nvidia.cpp. */
template <usearch_sequence_metric_t metric_>
struct sequence_nvidia_fatbin {
    static util::bytes_view_t bytes() noexcept;
};

#endif
#if USEARCH_ARCH_ROCM_

/** The library's code object of one kind's sequence kits for one AMD class, defined in that class's
 *  @c src/sequence/vram_variants_amd.hip. */
template <usearch_sequence_metric_t metric_, gpu_class_t gpu_class_>
struct sequence_amd_code_object {
    inline static constexpr nk_capability_t capability_k = numkong_capabilities_of(gpu_class_);
    static util::bytes_view_t bytes() noexcept;
};

#endif

namespace unum::usearch::test::sequence {
using namespace unum::usearch::sequence;

#if USEARCH_ARCH_CUDA_ || USEARCH_ARCH_ROCM_ || USEARCH_ARCH_METAL_

/** Draws @p count strings of about @p length nucleotides from eight templates, an eighth of every
 *  string's symbols substituted, inserted or erased. */
static std::vector<std::string> generate_nucleotides(std::size_t count, std::size_t length, u64_t seed) {
    constexpr std::string_view nucleotides_k = "ACGT";
    constexpr std::size_t templates_k = 8;
    split_mix_generator_t random_generator {seed};
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
        std::string packed(packed_codes_bytes(codes.size(), 2), '\0');
        pack_codes(codes, 2, bytes_of(packed.data(), packed.size()));
        symbols = std::move(packed);
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
static auto make_sequence_metric(std::size_t staging) {
    if constexpr (metric_ == usearch_sequence_metric_overlap_k) {
        overlap_metric<typename overlap_ladder<capability_>::kit_t> metric;
        expect_equal(metric.init(staging, vram_overlap_widths_k), status_code_t::success_k);
        return metric;
    }
    else if constexpr (metric_ == usearch_sequence_metric_needleman_wunsch_k) {
        needleman_wunsch_metric<typename needleman_wunsch_ladder<capability_>::kit_t> metric;
        expect_equal(metric.init(staging, vram_nucleotide_costs_k), status_code_t::success_k);
        return metric;
    }
    else {
        constexpr sz_levenshtein_symbol_t symbol_k = metric_ == usearch_sequence_metric_levenshtein_utf8_k
                                                         ? sz_levenshtein_runes_k
                                                         : sz_levenshtein_bytes_k;
        levenshtein_metric<typename levenshtein_ladder<symbol_k, capability_>::kit_t> metric;
        expect_equal(metric.init(staging), status_code_t::success_k);
        return metric;
    }
}

/** The kit the GPU scores @p metric_ with, its ladder's pick at @p capability_ , the CUDA floor
 *  unless named, or Metal's. */
template <usearch_sequence_metric_t metric_, sz_capability_t capability_ = sz_cap_cuda_k>
using vram_kit_of = std::conditional_t<
    metric_ == usearch_sequence_metric_overlap_k, typename overlap_ladder<capability_>::kit_t,
    std::conditional_t<
        metric_ == usearch_sequence_metric_needleman_wunsch_k, typename needleman_wunsch_ladder<capability_>::kit_t,
        typename levenshtein_ladder<metric_ == usearch_sequence_metric_levenshtein_utf8_k ? sz_levenshtein_runes_k
                                                                                          : sz_levenshtein_bytes_k,
                                    capability_>::kit_t>>;

#if USEARCH_ARCH_CUDA_

/** What the NVIDIA collections build on: the executor, the face over the CUDA floor's kits and the
 *  library's fatbins, its storage and the contexts carved from the executor's run. */
struct nvidia_cells_t {
    using executor_t = cuda_executor_t;
    template <usearch_sequence_metric_t metric_, typename node_id_type_>
    using index_t = vram_nvidia_index<vram_kit_of<metric_>, node_id_type_, sequence_nvidia_fatbin<metric_>>;

    static cuda_virtual_memory_allocator<byte_t> storage(executor_t& executor, int device) noexcept {
        return cuda_virtual_memory_allocator<byte_t> {executor.api(), device};
    }
    static cuda_arena_allocator_t arena(executor_t& executor) noexcept {
        return cuda_arena_allocator_t {executor.stream(0).arena, executor.stream(0)};
    }

    /** The sharded collections' shards, their storage on a device alone, and the runtime opening
     *  the executor. */
    template <typename index_type_, typename build_type_>
    using indexes_t = us::basic::vram_nvidia_indexes<index_type_, util::uuid128_t, build_type_>;
    static cuda_virtual_memory_allocator<byte_t> storage(int device) noexcept {
        return cuda_virtual_memory_allocator<byte_t> {test::gpu_drivers().cuda, device};
    }
    static cuda_api_t const& api() noexcept { return test::gpu_drivers().cuda; }
    inline static constexpr gpu_vendor_t vendor_k = gpu_vendor_t::nvidia_k;
    inline static constexpr char runtime_k[] = "CUDA";
};
using cell_vendor_t = nvidia_cells_t;

#endif
#if USEARCH_ARCH_ROCM_

/** What the AMD collections build on: the executor, the face over the CDNA3 class's kits, whose
 *  64-lane wavefronts CDNA3 and CDNA4 run, and the library's code objects, its storage and the
 *  contexts carved from the executor's run. */
struct amd_cells_t {
    using executor_t = hip_executor_t;
    template <usearch_sequence_metric_t metric_, typename node_id_type_>
    using index_t = vram_amd_index<vram_kit_of<metric_, stringzilla_capabilities_of(gpu_class_t::amd_cdna3_k)>,
                                   node_id_type_, sequence_amd_code_object<metric_, gpu_class_t::amd_cdna3_k>>;

    static hip_virtual_memory_allocator<byte_t> storage(executor_t& executor, int device) noexcept {
        return hip_virtual_memory_allocator<byte_t> {executor.api(), device};
    }
    static hip_arena_allocator_t arena(executor_t& executor) noexcept {
        return hip_arena_allocator_t {executor.stream(0).arena, executor.stream(0)};
    }

    /** The sharded collections' shards, their storage on a device alone, and the runtime opening
     *  the executor. */
    template <typename index_type_, typename build_type_>
    using indexes_t = us::basic::vram_amd_indexes<index_type_, util::uuid128_t, build_type_>;
    static hip_virtual_memory_allocator<byte_t> storage(int device) noexcept {
        return hip_virtual_memory_allocator<byte_t> {test::gpu_drivers().hip, device};
    }
    static hip_api_t const& api() noexcept { return test::gpu_drivers().hip; }
    inline static constexpr gpu_vendor_t vendor_k = gpu_vendor_t::amd_k;
    inline static constexpr char runtime_k[] = "ROCm";
};
using cell_vendor_t = amd_cells_t;

#endif
#if USEARCH_ARCH_METAL_ && !USEARCH_ARCH_CUDA_

/** What the Apple collections build on: the executor, the face over Metal's kits, its storage and
 *  the contexts over its stream. */
struct apple_cells_t {
    using executor_t = metal_executor_t;
    template <usearch_sequence_metric_t metric_, typename node_id_type_>
    using index_t = vram_apple_index<vram_kit_of<metric_, sz_cap_metal_k>, node_id_type_>;

    static metal_shared_allocator_t storage(executor_t& executor, int) noexcept {
        return metal_shared_allocator_t {executor.stream(0).handle};
    }
    static metal_stream_t arena(executor_t& executor) noexcept { return executor.stream(0).handle.share(); }
};
using cell_vendor_t = apple_cells_t;

#endif

/** Matches, of @p wanted at most, whose distances reach no further than the @p wanted -th exact
 *  one: ties make the IDs ambiguous, the distances never. */
static std::size_t matched_count(std::span<sz_size_t const> found, nk::vector_view<sz_size_t> exact,
                                 std::size_t wanted) {
    nk::vector<sz_size_t> ranked = make_vector<sz_size_t>(exact.size());
    std::memcpy(ranked.data(), exact.data(), exact.size() * sizeof(sz_size_t));
    std::span<sz_size_t> const nearest = to_writable_span(ranked);
    std::nth_element(nearest.begin(), nearest.begin() + (wanted - 1), nearest.end());
    std::size_t matched = 0;
    for (sz_size_t const distance : found) matched += distance <= nearest[wanted - 1];
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
 *  Every match the GPU reports must hold the pair's exact distance, ascending, and under
 *  @p reference the GPU's recall at ten must reach the RAM index's at the same breadth, give or
 *  take a twentieth.
 */
template <usearch_sequence_metric_t metric_, typename node_id_type_ = u32_t>
static void test_sequence_vram_collection(u64_t seed, cell_vendor_t::executor_t& executor, std::size_t count,
                                          std::size_t length, std::size_t queries_count,
                                          recall_reference_t reference = recall_reference_t::ram_index_k) {
    using ram_index_t = ram_index<util::uuid128_t, u32_t>;
    using vram_index_t = typename cell_vendor_t::template index_t<metric_, node_id_type_>;
    constexpr std::size_t wanted_k = 10, expansion_k = 64, batches_k = 3, origin_k = 3;
    checks_held_t const held;

    std::vector<std::string> strings = generate_nucleotides(count + queries_count, length, seed);
    if constexpr (metric_ == usearch_sequence_metric_levenshtein_utf8_k) respell_as_runes(strings);
    if constexpr (metric_ == usearch_sequence_metric_needleman_wunsch_k) pack_as_codes(strings);
    std::string tape(origin_k, '\0');
    std::vector<std::size_t> offsets {origin_k};
    for (std::string const& symbols : strings) tape += symbols, offsets.push_back(tape.size());
    byte_t const* const tape_bytes = reinterpret_cast<byte_t const*>(tape.data());

    // The exact distances of every query to every stored string, through the serial kit.
    nk::matrix<sz_size_t> exact = make_matrix<sz_size_t>(queries_count, count);
    auto serial_metric = make_sequence_metric<metric_, sz_cap_serial_k>(count);
    for (std::size_t query = 0; query != queries_count; ++query) {
        serial_metric.reset(std::string_view {strings[count + query]});
        for (std::size_t entry = 0; entry != count; ++entry) serial_metric.push(std::string_view {strings[entry]});
        nk::vector_span<sz_size_t> const distances = exact.span().row(query).as_vector();
        serial_metric.drain(to_writable_span(distances), us::basic::unbounded_radius<sz_size_t>());
    }

    // The RAM index over the same sequences, at the same breadth, which an exact-distance reference
    // leaves unbuilt and its recall at zero.
    std::size_t ram_matched = 0;
    if (reference == recall_reference_t::ram_index_k) {
        ram_index_t ram_index;
        auto metric = make_sequence_metric<metric_, sz_cap_cpus_k>(expansion_k * connectivity_32_k);
        ram_index_t::traversal_context_t ram_context {seed};
        expect_true(bool(ram_index.reserve(count)));
        for (std::size_t entry = 0; entry != count; ++entry) {
            ram_index_t::value_t value;
            [[maybe_unused]] sz::status_t const assigned = value.assign(strings[entry]);
            util::uuid128_t const key = uuid_of(entry);
            expect_equal(ram_index
                             .insert({&value, 1}, {&key, 1}, metric, {}, ram_context,
                                     us::basic::keep_diverse_t {connectivity_32_k}, {expansion_k})
                             .status,
                         status_code_t::success_k);
        }
        for (std::size_t query = 0; query != queries_count; ++query) {
            expect_equal(
                ram_index.search(std::string_view {strings[count + query]}, {expansion_k}, metric, ram_context).status,
                status_code_t::success_k);
            std::vector<sz_size_t> found;
            for (std::size_t rank = 0; rank != min_of(wanted_k, ram_context.results.size()); ++rank)
                found.push_back(ram_context.results[rank].distance);
            ram_matched += matched_count(found, exact.view().row(query).as_vector(), wanted_k);
        }
    }

    // The GPU index, in three batches, each over a context carved from the executor's run.
    int const device = executor.device(0);
    vram_index_t vram_index {cell_vendor_t::storage(executor, device)};
    if constexpr (metric_ == usearch_sequence_metric_overlap_k) {
        u32_t widths[std::size(vram_overlap_widths_k)];
        for (std::size_t window = 0; window != std::size(widths); ++window)
            widths[window] = static_cast<u32_t>(vram_overlap_widths_k[window]);
        expect_equal(vram_index.parameters(widths), status_code_t::success_k);
    }
    if constexpr (metric_ == usearch_sequence_metric_needleman_wunsch_k)
        expect_equal(vram_index.parameters(vram_nucleotide_costs_k), status_code_t::success_k);
    expect_equal(vram_index.reserve({count, connectivity_32_k, offsets[count] - offsets[0]}).status,
                 status_code_t::success_k);
    typename vram_index_t::insert_config_t const insert_config {expansion_k};
    nk::vector<util::uuid128_t> keys = make_vector<util::uuid128_t>(count);
    for (std::size_t entry = 0; entry != count; ++entry) keys[entry] = uuid_of(entry);
    for (std::size_t batch = 0; batch != batches_k && held(); ++batch) {
        std::size_t const first = count * batch / batches_k, bound = count * (batch + 1) / batches_k;
        std::span<std::size_t const> const batch_offsets {offsets.data() + first, bound - first + 1};
        expect_equal(executor.reserve(vram_index.traversal_memory_usage(
                         tape_view<byte_t const> {tape_bytes, batch_offsets}, insert_config)),
                     status_code_t::success_k);
        typename vram_index_t::traversal_context_t context {cell_vendor_t::arena(executor)};
        auto const [inserted, status] = vram_index.insert({keys.data() + first, bound - first},
                                                          {tape_bytes, batch_offsets}, context, insert_config);
        expect_equal(status, status_code_t::success_k);
        expect_equal(inserted.first_id, first);
        expect_equal(inserted.new_size, bound);
        expect_true(inserted.computed_distances > 0);
    }
    expect_equal(vram_index.size(), count);

    nk::matrix<node_id_type_> ids = make_matrix<node_id_type_>(queries_count, wanted_k);
    nk::matrix<f32_t> distances = make_matrix<f32_t>(queries_count, wanted_k);
    std::span<std::size_t const> const query_offsets {offsets.data() + count, queries_count + 1};
    typename vram_index_t::search_config_t const search_config {expansion_k, wanted_k};
    expect_equal(executor.reserve(vram_index.traversal_memory_usage(tape_view<byte_t const> {tape_bytes, query_offsets},
                                                                    search_config)),
                 status_code_t::success_k);
    typename vram_index_t::traversal_context_t context {cell_vendor_t::arena(executor)};
    if (held())
        expect_equal(
            vram_index.search({tape_bytes, query_offsets}, ids.span(), distances.span(), search_config, context).status,
            status_code_t::success_k);
    std::size_t vram_matched = 0;
    error_stats_t distance_errors;
    for (std::size_t query = 0; query != queries_count && held(); ++query) {
        std::vector<sz_size_t> found;
        for (std::size_t rank = 0; rank != wanted_k; ++rank) {
            node_id_type_ const id = ids(query, rank);
            if (id == (std::numeric_limits<node_id_type_>::max)()) break;
            // The candidates keep 31 bits of a float, so the overlap kind's fixed point loses a
            // mantissa bit more on the device.
            f32_t const distance = distances(query, rank);
            if (!expect_true(id < count)) break;
            distance_errors.add(distance, static_cast<f32_t>(exact(query, id)));
            expect_true(found.empty() || found.back() <= exact(query, id));
            auto const [stored_key, key_status] = vram_index.key_of(id);
            expect_true(succeeded(key_status) && stored_key == keys[id]);
            found.push_back(exact(query, id));
        }
        expect_equal(found.size(), wanted_k);
        vram_matched += matched_count(found, exact.view().row(query).as_vector(), wanted_k);
    }
    f64_t const ram_recall = static_cast<f64_t>(ram_matched) / (queries_count * wanted_k);
    f64_t const vram_recall = static_cast<f64_t>(vram_matched) / (queries_count * wanted_k);
    if (vram_recall + 0.05 < ram_recall)
        fmt::print(stderr, "  recall at ten on {} strings of {} bytes: {:.3f} on the GPU, {:.3f} in RAM\n", count,
                   length, vram_recall, ram_recall);
    expect_true(vram_recall + 0.05 >= ram_recall);
    expect_errors("vram sequence distances", comparison_family_t::approximate_k, comparison_verdict_t::assert_k,
                  distance_errors, 0);
}

#endif // USEARCH_ARCH_CUDA_ || USEARCH_ARCH_ROCM_ || USEARCH_ARCH_METAL_

#if USEARCH_ARCH_CUDA_ || USEARCH_ARCH_ROCM_

/** Builds the byte Levenshtein shards of a sharded index, unreserved, on the device it names, over
 *  the cells of @p cells_type_ 's vendor, which keeps each vendor's builder a symbol of its own. */
template <typename cells_type_>
struct sequence_vram_shards {
    using index_t = typename cells_type_::template index_t<usearch_sequence_metric_levenshtein_bytes_k, u32_t>;

    std::unique_ptr<index_t> operator()(int device) const noexcept {
        return std::unique_ptr<index_t>(new (std::nothrow) index_t {cells_type_::storage(device)});
    }
};
using sequence_vram_shards_t = sequence_vram_shards<cell_vendor_t>;

/** A checkpoint of @p saved loads back from its files through the staging slots, its neighbors and
 *  values converted on the device, keeps every key, and saves again byte for byte. */
template <typename indexes_type_>
static void expect_sequence_vram_reloaded(indexes_type_& saved, std::size_t shard_capacity,
                                          cell_vendor_t::executor_t& executor) {
    scratch_directory_t const scratch;
    std::string const first_path = scratch.path("saved.safegraphs");
    std::string const second_path = scratch.path("resaved.safegraphs");
    save_config_t config;
    config.synchronization = synchronization_t::immutable_k;
    expect_equal(saved.save(first_path, codec_t {}, executor, config, "levenshtein"), status_code_t::success_k);
    indexes_type_ loaded {forkunion_allocator_t {no_memory_domain_k}, sequence_vram_shards_t {}};
    loaded.bound_shards(shard_capacity);
    expect_equal(loaded.emplace(executor.stream(0).device), status_code_t::success_k);
    expect_equal(loaded.load(std::string_view {first_path}, codec_t {}, executor, load_config_t {}, connectivity_32_k),
                 status_code_t::success_k);
    expect_equal(loaded.size(), saved.size());
    expect_equal(loaded.save(second_path, codec_t {}, executor, config, "levenshtein"), status_code_t::success_k);
    std::size_t const parts = saved.shards_count();
    growing_buffer<char> first_part {global_allocator}, second_part {global_allocator};
    for (std::size_t part = 0; part != parts; ++part) {
        auto const [first_named, first_status] = safegraphs_part_path(first_part, safegraphs_stem(first_path), part,
                                                                      parts);
        auto const [second_named, second_status] = safegraphs_part_path(second_part, safegraphs_stem(second_path), part,
                                                                        parts);
        expect_true(succeeded(first_status) && succeeded(second_status));
        std::string const first_name {parts == 1 ? std::string_view {first_path} : first_named};
        std::string const second_name {parts == 1 ? std::string_view {second_path} : second_named};
        auto const [first_mapping, first_map_status] = mapped_file_t::read(first_name);
        auto const [second_mapping, second_map_status] = mapped_file_t::read(second_name);
        util::bytes_view_t const first_bytes = first_mapping.view(), second_bytes = second_mapping.view();
        expect_true(succeeded(first_map_status) && succeeded(second_map_status) &&
                    first_bytes.size() == second_bytes.size() &&
                    std::memcmp(first_bytes.data(), second_bytes.data(), first_bytes.size()) == 0);
        std::remove(first_name.c_str()), std::remove(second_name.c_str());
    }
    std::remove(first_path.c_str()), std::remove(second_path.c_str());
}

/** Two shards on one device bounded at 1000 entries take 5000 sequences in one batch, which opens
 *  two more generations of two: every key counts once across the six shards, and the merged recall
 *  at ten reaches one unbounded shard's, give or take a twentieth. That shard reserves no capacity,
 *  so its keys cover as its nodes map, and each counts once in it too. */
static void test_sequence_vram_sharded(u64_t seed, cell_vendor_t::executor_t& executor) {
    using index_t = sequence_vram_shards_t::index_t;
    using indexes_t = cell_vendor_t::indexes_t<index_t, sequence_vram_shards_t>;
    constexpr std::size_t entries_k = 5000, shard_capacity_k = 1000, queries_k = 64, wanted_k = 10, expansion_k = 64;

    std::vector<std::string> const strings = generate_nucleotides(entries_k + queries_k, 100, seed);
    std::string tape;
    std::vector<std::size_t> offsets {0};
    for (std::string const& symbols : strings) tape += symbols, offsets.push_back(tape.size());
    byte_t const* const tape_bytes = reinterpret_cast<byte_t const*>(tape.data());
    tape_view<byte_t const> const sequences {tape_bytes, {offsets.data(), entries_k + 1}};
    tape_view<byte_t const> const queries {tape_bytes, {offsets.data() + entries_k, queries_k + 1}};
    nk::matrix<sz_size_t> exact = make_matrix<sz_size_t>(queries_k, entries_k);
    auto serial_metric = make_sequence_metric<usearch_sequence_metric_levenshtein_bytes_k, sz_cap_serial_k>(entries_k);
    for (std::size_t query = 0; query != queries_k; ++query) {
        serial_metric.reset(std::string_view {strings[entries_k + query]});
        for (std::size_t entry = 0; entry != entries_k; ++entry) serial_metric.push(std::string_view {strings[entry]});
        nk::vector_span<sz_size_t> const distances = exact.span().row(query).as_vector();
        serial_metric.drain(to_writable_span(distances), us::basic::unbounded_radius<sz_size_t>());
    }
    nk::vector<util::uuid128_t> keys = make_vector<util::uuid128_t>(entries_k);
    for (std::size_t entry = 0; entry != entries_k; ++entry) keys[entry] = uuid_of(entry);

    indexes_t indexes {forkunion_allocator_t {no_memory_domain_k}, sequence_vram_shards_t {}};
    indexes.bound_shards(shard_capacity_k);
    expect_equal(indexes.emplace(executor.stream(0).device), status_code_t::success_k);
    expect_equal(indexes.emplace(executor.stream(0).device), status_code_t::success_k);
    expect_equal(indexes.reserve({entries_k, connectivity_32_k, offsets[entries_k]}).status, status_code_t::success_k);
    auto const [inserted, insert_status] = indexes.insert(to_span(keys), sequences, executor, {expansion_k});
    expect_equal(insert_status, status_code_t::success_k);
    expect_true(inserted.computed_distances > 0);
    expect_equal(indexes.shards_count(), std::size_t {6});
    expect_equal(indexes.size(), entries_k);
    std::size_t miscounted = 0;
    for (util::uuid128_t const& key : keys) miscounted += indexes.count(key) != 1;
    expect_equal(miscounted, std::size_t {0});
    auto const [health, health_status] = indexes.health(executor);
    expect_equal(health_status, status_code_t::success_k);
    expect_equal(health.audited_nodes, entries_k);
    expect_equal(health.reachable_nodes, entries_k);

    nk::matrix<util::uuid128_t> found_keys = make_matrix<util::uuid128_t>(queries_k, wanted_k);
    nk::matrix<f32_t> found_distances = make_matrix<f32_t>(queries_k, wanted_k);
    nk::vector<std::size_t> counts = make_vector<std::size_t>(queries_k);
    typename index_t::search_config_t const search_config {expansion_k, wanted_k};
    expect_equal(indexes
                     .search(queries, found_keys.span(), found_distances.span(), to_writable_span(counts),
                             search_config, executor)
                     .status,
                 status_code_t::success_k);
    std::size_t sharded_matched = 0;
    for (std::size_t query = 0; query != queries_k; ++query) {
        std::vector<sz_size_t> found;
        for (std::size_t rank = 0; rank != counts[query]; ++rank)
            found.push_back(static_cast<sz_size_t>(found_distances(query, rank)));
        sharded_matched += matched_count(found, exact.view().row(query).as_vector(), wanted_k);
    }

    index_t single {cell_vendor_t::storage(executor, executor.device(0))};
    expect_equal(single.reserve({0, connectivity_32_k, 0}).status, status_code_t::success_k);
    typename index_t::insert_config_t const insert_config {expansion_k};
    expect_equal(executor.reserve(max_of(single.traversal_memory_usage(sequences, insert_config),
                                         single.traversal_memory_usage(queries, search_config))),
                 status_code_t::success_k);
    typename index_t::traversal_context_t context {cell_vendor_t::arena(executor)};
    expect_equal(single.insert(to_span(keys), sequences, context, insert_config).status, status_code_t::success_k);
    for (util::uuid128_t const& key : keys) miscounted += single.count(key) != 1;
    expect_equal(miscounted, std::size_t {0});
    nk::matrix<u32_t> ids = make_matrix<u32_t>(queries_k, wanted_k);
    expect_equal(single.search(queries, ids.span(), found_distances.span(), search_config, context).status,
                 status_code_t::success_k);
    std::size_t single_matched = 0;
    for (std::size_t query = 0; query != queries_k; ++query) {
        std::vector<sz_size_t> found;
        for (std::size_t rank = 0; rank != wanted_k && ids(query, rank) != (std::numeric_limits<u32_t>::max)(); ++rank)
            found.push_back(static_cast<sz_size_t>(found_distances(query, rank)));
        single_matched += matched_count(found, exact.view().row(query).as_vector(), wanted_k);
    }
    expect_true(sharded_matched + rewrite_recall_margin_k * queries_k * wanted_k >= single_matched);
    expect_sequence_vram_reloaded(indexes, shard_capacity_k, executor);
}

/** Rounds over a sharded index of two shards: a refine plan driven to its end concludes a real
 *  verdict and fills the round's report, a next round runs over the same plan, and a finished plan
 *  answers @c uninitialized_k to another run or finish. Two sources gather and cut into two targets
 *  of two shards each, then one target cuts back into the emptied sources, every entry landing
 *  exactly once and every source ending empty, after refusals left every index whole: empty or
 *  repeated lists, a shard capacity, one other index that is neither a gather into this one nor a
 *  cut of it, and a target of one shard, refused before the other source moves in. */
static void test_sequence_vram_rounds(u64_t seed, cell_vendor_t::executor_t& executor) {
    using index_t = sequence_vram_shards_t::index_t;
    using indexes_t = cell_vendor_t::indexes_t<index_t, sequence_vram_shards_t>;
    constexpr std::size_t entries_k = 512, lone_k = 64, expansion_k = 64;

    std::vector<std::string> const strings = generate_nucleotides(entries_k, 100, seed);
    std::string tape;
    std::vector<std::size_t> offsets {0};
    for (std::string const& symbols : strings) tape += symbols, offsets.push_back(tape.size());
    tape_view<byte_t const> const sequences {reinterpret_cast<byte_t const*>(tape.data()),
                                             {offsets.data(), entries_k + 1}};
    nk::vector<util::uuid128_t> keys = make_vector<util::uuid128_t>(entries_k);
    for (std::size_t entry = 0; entry != entries_k; ++entry) keys[entry] = uuid_of(entry);

    auto placed = [&]() {
        auto indexes = std::make_unique<indexes_t>(forkunion_allocator_t {no_memory_domain_k},
                                                   sequence_vram_shards_t {});
        expect_equal(indexes->emplace(executor.stream(0).device), status_code_t::success_k);
        expect_equal(indexes->emplace(executor.stream(0).device), status_code_t::success_k);
        return indexes;
    };
    std::unique_ptr<indexes_t> const source = placed(), first = placed(), second = placed(), lone = placed();
    expect_equal(source->reserve({entries_k, connectivity_32_k, offsets[entries_k]}).status, status_code_t::success_k);
    expect_equal(source->insert(to_span(keys), sequences, executor, {expansion_k}).status, status_code_t::success_k);
    auto narrow = std::make_unique<indexes_t>(forkunion_allocator_t {no_memory_domain_k}, sequence_vram_shards_t {});
    expect_equal(narrow->emplace(executor.stream(0).device), status_code_t::success_k);

    // The second source holds keys of its own, so a refusal that moved it would show in its size.
    std::vector<std::string> const lone_strings = generate_nucleotides(lone_k, 100, seed + 1);
    std::string lone_tape;
    std::vector<std::size_t> lone_offsets {0};
    for (std::string const& symbols : lone_strings) lone_tape += symbols, lone_offsets.push_back(lone_tape.size());
    tape_view<byte_t const> const lone_sequences {reinterpret_cast<byte_t const*>(lone_tape.data()),
                                                  {lone_offsets.data(), lone_k + 1}};
    nk::vector<util::uuid128_t> lone_keys = make_vector<util::uuid128_t>(lone_k);
    for (std::size_t entry = 0; entry != lone_k; ++entry) lone_keys[entry] = uuid_of(entries_k + entry);
    expect_equal(lone->reserve({lone_k, connectivity_32_k, lone_offsets[lone_k]}).status, status_code_t::success_k);
    expect_equal(lone->insert(to_span(lone_keys), lone_sequences, executor, {expansion_k}).status,
                 status_code_t::success_k);

    {
        auto [plan, plan_status] = source->plan_refine({}, executor);
        expect_equal(plan_status, status_code_t::success_k);
        expect_equal(plan.run(executor), status_code_t::success_k);
        util::refine_report_t const round = plan.running();
        expect_true(round.verdict == util::refine_verdict_t::improved_k ||
                    round.verdict == util::refine_verdict_t::converged_k);
        expect_true(round.neighborhoods != 0 && round.opening.audited_nodes == entries_k);
        plan.next_round();
        expect_equal(plan.run(executor), status_code_t::success_k);
        expect_equal(plan.finish(executor).status, status_code_t::success_k);
        expect_equal(plan.run(executor), status_code_t::uninitialized_k);
        expect_equal(plan.finish(executor).status, status_code_t::uninitialized_k);
    }

    using listed_t = std::span<indexes_t* const>;
    indexes_t* const sources[1] = {source.get()};
    indexes_t* const targets[2] = {first.get(), second.get()};
    indexes_t* const repeated[2] = {first.get(), first.get()};
    indexes_t* const several[2] = {source.get(), lone.get()};
    indexes_t* const one[1] = {lone.get()};
    indexes_t* const uneven[2] = {first.get(), narrow.get()};
    expect_equal(source->plan_reshard(listed_t {sources}, listed_t {}, {}, executor).status,
                 status_code_t::invalid_argument_k);
    expect_equal(source->plan_reshard(listed_t {sources}, listed_t {repeated}, {}, executor).status,
                 status_code_t::invalid_argument_k);
    expect_equal(source->plan_reshard(listed_t {sources}, listed_t {targets}, {{}, 16, 0}, executor).status,
                 status_code_t::unsupported_k);
    expect_equal(source->plan_reshard(listed_t {sources}, listed_t {one}, {}, executor).status,
                 status_code_t::config_mismatch_k);
    expect_equal(source->plan_reshard(listed_t {several}, listed_t {uneven}, {}, executor).status,
                 status_code_t::config_mismatch_k);
    expect_equal(source->size(), entries_k);
    expect_equal(lone->size(), lone_k);

    auto const [cut, cut_status] = source->reshard(listed_t {several}, listed_t {targets}, {}, executor);
    expect_equal(cut_status, status_code_t::success_k);
    expect_true(cut.verdict != util::refine_verdict_t::none_k && cut.neighborhoods != 0);
    expect_equal(source->size() + lone->size(), std::size_t {0});
    expect_equal(first->size() + second->size(), entries_k + lone_k);

    // The first target cuts back into the emptied sources, one source into several targets.
    indexes_t* const back[1] = {first.get()};
    auto const [recut, recut_status] = first->reshard(listed_t {back}, listed_t {several}, {}, executor);
    expect_equal(recut_status, status_code_t::success_k);
    expect_equal(first->size(), std::size_t {0});
    expect_equal(source->size() + lone->size() + second->size(), entries_k + lone_k);
    std::size_t miscounted = 0;
    auto const landed = [&](util::uuid128_t const& key) {
        return source->count(key) + lone->count(key) + second->count(key);
    };
    for (util::uuid128_t const& key : keys) miscounted += landed(key) != 1;
    for (util::uuid128_t const& key : lone_keys) miscounted += landed(key) != 1;
    expect_equal(miscounted, std::size_t {0});
}

/** A reshard of the sharded index into one other index, neither a gather into it nor a cut of it,
 *  is refused and leaves it whole: its size and the count of every key stay as inserted. */
static void test_sequence_vram_cut_refused(u64_t seed, cell_vendor_t::executor_t& executor) {
    using index_t = sequence_vram_shards_t::index_t;
    using indexes_t = cell_vendor_t::indexes_t<index_t, sequence_vram_shards_t>;
    constexpr std::size_t entries_k = 256, expansion_k = 64;

    std::vector<std::string> const strings = generate_nucleotides(entries_k, 100, seed);
    std::string tape;
    std::vector<std::size_t> offsets {0};
    for (std::string const& symbols : strings) tape += symbols, offsets.push_back(tape.size());
    tape_view<byte_t const> const sequences {reinterpret_cast<byte_t const*>(tape.data()),
                                             {offsets.data(), entries_k + 1}};
    nk::vector<util::uuid128_t> keys = make_vector<util::uuid128_t>(entries_k);
    for (std::size_t entry = 0; entry != entries_k; ++entry) keys[entry] = uuid_of(entry);

    indexes_t source {forkunion_allocator_t {no_memory_domain_k}, sequence_vram_shards_t {}};
    indexes_t lone {forkunion_allocator_t {no_memory_domain_k}, sequence_vram_shards_t {}};
    for (indexes_t* placed : {&source, &lone}) {
        expect_equal(placed->emplace(executor.stream(0).device), status_code_t::success_k);
        expect_equal(placed->emplace(executor.stream(0).device), status_code_t::success_k);
    }
    expect_equal(source.reserve({entries_k, connectivity_32_k, offsets[entries_k]}).status, status_code_t::success_k);
    expect_equal(source.insert(to_span(keys), sequences, executor, {expansion_k}).status, status_code_t::success_k);

    indexes_t* const sources[1] = {&source};
    indexes_t* const targets[1] = {&lone};
    expect_equal(
        source.plan_reshard(std::span<indexes_t* const> {sources}, std::span<indexes_t* const> {targets}, {}, executor)
            .status,
        status_code_t::config_mismatch_k);
    expect_equal(source.size(), entries_k);
    std::size_t miscounted = 0;
    for (util::uuid128_t const& key : keys) miscounted += source.count(key) != 1;
    expect_equal(miscounted, std::size_t {0});
}

/** Sequences inserted into a GPU handle through the C interface come back from its @c get byte for
 *  byte, each key's tape gathered off the device. */
static void test_sequence_vram_get(u64_t seed, std::int32_t ordinal) {
    constexpr std::size_t entries_k = 8;
    checks_held_t const held;

    std::vector<std::string> const strings = generate_nucleotides(entries_k, 100, seed);
    std::string tape;
    std::vector<std::size_t> offsets {0};
    for (std::string const& symbols : strings) tape += symbols, offsets.push_back(tape.size());
    nk::vector<usearch_key_t> keys = make_vector<usearch_key_t>(entries_k);
    for (std::size_t entry = 0; entry != entries_k; ++entry) keys[entry] = us::tools::key_of_offset(seed, entry);

    usearch_outcome_t outcome {};
    usearch_arena_t arena = nullptr;
    usearch_arena_init_t arena_init {};
    arena_init.arena = &arena;
    arena_init.outcome = &outcome;
    usearch_arena_init(&arena_init);
    expect_true(!outcome.message);

    std::int32_t const devices[] = {ordinal};
    usearch_vram_options_t const vram {devices, 1};
    usearch_sequence_index_t index = nullptr;
    usearch_sequence_init_vram_t init {};
    init.index = &index;
    init.outcome = &outcome;
    init.metric = usearch_sequence_metric_levenshtein_bytes_k;
    init.vram = vram;
    if (held()) usearch_sequence_init_vram(&init);
    expect_true(!outcome.message);
    usearch_executor_t executor = nullptr;
    usearch_executor_for_vram_t placement {};
    placement.executor = &executor;
    placement.outcome = &outcome;
    placement.vram = vram;
    if (held()) usearch_executor_for_vram(&placement);
    expect_true(!outcome.message);

    usearch_sequence_insert_t insert {};
    insert.index = index;
    insert.executor = executor;
    insert.outcome = &outcome;
    insert.count = entries_k;
    insert.keys = {keys.data(), sizeof(usearch_key_t)};
    insert.sequences = tools::tape_of(tape.data(), offsets.data(), usearch_scalar_u8_k);
    if (held()) usearch_sequence_insert(&insert);
    expect_true(!outcome.message);

    for (std::size_t entry = 0; entry != entries_k && held(); ++entry) {
        usearch_tape_view_t got {};
        std::size_t written = 0;
        usearch_sequence_get_t get {};
        get.index = index;
        get.executor = executor;
        get.outcome = &outcome;
        get.key = keys[entry];
        get.sequences = &got;
        get.written = &written;
        get.arena = arena;
        usearch_sequence_get(&get);
        expect_true(!outcome.message);
        if (outcome.message) break;
        expect_equal(written, std::size_t {1});
        expect_true(written == 1 && tools::tape_offset(got, 1) - tools::tape_offset(got, 0) == strings[entry].size() &&
                    std::memcmp(static_cast<char const*>(got.data) + tools::tape_offset(got, 0), strings[entry].data(),
                                strings[entry].size()) == 0);
    }

    usearch_sequence_free(index);
    usearch_executor_free(executor);
    usearch_arena_free(arena);
}

template <>
void run_sequence_vram_cases<cell_vendor_t::executor_t>(gpu_device_t const& device) {
    if (device.vendor != cell_vendor_t::vendor_k) return;
    char const* const vendor = constant_of(cell_vendor_t::vendor_k);
#if USEARCH_ARCH_ROCM_ && !USEARCH_TARGET_AMD_CDNA3
    return run_test_case(fmt::format("run_sequence_vram_cases({})", vendor).c_str(),
                         [] { return case_result_t::skipped(built_without_cdna3_k); });
#else
    // The AMD cells' kits run the CDNA3 class's 64-lane wavefronts, which a CDNA5 device lacks.
    if (cell_vendor_t::vendor_k == gpu_vendor_t::amd_k && device.warp_lanes != 64)
        return run_test_case(fmt::format("run_sequence_vram_cases({})", vendor).c_str(),
                             [] { return case_result_t::skipped(needs_64_lane_waves_k); });
    int const ordinal = static_cast<int>(device.vendor_ordinal);
    auto [executor, spawn_status] = cell_vendor_t::executor_t::spawn(cell_vendor_t::api(),
                                                                     std::span<int const> {&ordinal, 1});
    if (!expect_true(succeeded(spawn_status), "{} device #{} opens no stream", cell_vendor_t::runtime_k, ordinal))
        return;
    u64_t const seed = test_run.settings.seed;
    auto const run = [&](char const* call, std::string_view arguments, auto&& check) {
        if (arguments.empty()) run_test_case(fmt::format("{}({})", call, vendor).c_str(), check);
        else run_test_case(fmt::format("{}({}, {})", call, vendor, arguments).c_str(), check);
    };
    constexpr usearch_sequence_metric_t bytes_k = usearch_sequence_metric_levenshtein_bytes_k;
    constexpr usearch_sequence_metric_t runes_k = usearch_sequence_metric_levenshtein_utf8_k;
    constexpr usearch_sequence_metric_t overlap_k = usearch_sequence_metric_overlap_k;
    constexpr usearch_sequence_metric_t aligned_k = usearch_sequence_metric_needleman_wunsch_k;
    run("test_sequence_vram_collection<bytes_k>", "2048, 100, 128",
        [&] { test_sequence_vram_collection<bytes_k>(stream_key(seed, "short"), executor, 2048, 100, 128); });
    run("test_sequence_vram_collection<bytes_k>", "512, 200, 64",
        [&] { test_sequence_vram_collection<bytes_k>(stream_key(seed, "long"), executor, 512, 200, 64); });
    run("test_sequence_vram_collection<bytes_k>", "256, 1100, 32",
        [&] { test_sequence_vram_collection<bytes_k>(stream_key(seed, "block"), executor, 256, 1100, 32); });
    run("test_sequence_vram_collection<runes_k>", "512, 200, 64",
        [&] { test_sequence_vram_collection<runes_k>(stream_key(seed, "runes"), executor, 512, 200, 64); });
    run("test_sequence_vram_collection<overlap_k>", "512, 200, 64",
        [&] { test_sequence_vram_collection<overlap_k>(stream_key(seed, "overlap"), executor, 512, 200, 64); });
    run("test_sequence_vram_collection<aligned_k>", "512, 200, 64",
        [&] { test_sequence_vram_collection<aligned_k>(stream_key(seed, "aligned"), executor, 512, 200, 64); });
    // Strided alignments past one tile only need to match the exact costs; a host index over them
    // would spend its minutes on diversity pruning.
    run("test_sequence_vram_collection<aligned_k>", "256, 1100, 8", [&] {
        test_sequence_vram_collection<aligned_k>(stream_key(seed, "aligned_stripes"), executor, 256, 1100, 8,
                                                 recall_reference_t::exact_distances_k);
    });
    run("test_sequence_vram_sharded", "", [&] { test_sequence_vram_sharded(stream_key(seed, "plural"), executor); });
    run("test_sequence_vram_rounds", "", [&] { test_sequence_vram_rounds(stream_key(seed, "rounds"), executor); });
    run("test_sequence_vram_cut_refused", "",
        [&] { test_sequence_vram_cut_refused(stream_key(seed, "cut_refused"), executor); });
    run("test_sequence_vram_get", "", [&] {
        // The C interface names devices by their place in the listing, which `device` belongs to.
        test_sequence_vram_get(stream_key(seed, "get"),
                               static_cast<std::int32_t>(&device - test::gpu_devices().data()));
    });
#endif
}

#elif USEARCH_ARCH_METAL_

/** The Apple cells: every kind the NVIDIA collections check, on the same data, judged the same. */
template <>
void run_sequence_vram_cases<metal_executor_t>(gpu_device_t const& device) {
    if (device.vendor != gpu_vendor_t::apple_k) return;
    int const ordinal = static_cast<int>(device.vendor_ordinal);
    auto [executor, spawn_status] = metal_executor_t::spawn(test::gpu_drivers().metal_contexts,
                                                            std::span<int const> {&ordinal, 1});
    if (!expect_true(succeeded(spawn_status), "Metal device #{} opens no stream", ordinal)) return;
    u64_t const seed = test_run.settings.seed;
    auto const run = [&](char const* call, std::string_view arguments, auto&& check) {
        run_test_case(fmt::format("{}(gpu_vendor_t::apple_k, {})", call, arguments).c_str(), check);
    };
    constexpr usearch_sequence_metric_t bytes_k = usearch_sequence_metric_levenshtein_bytes_k;
    constexpr usearch_sequence_metric_t runes_k = usearch_sequence_metric_levenshtein_utf8_k;
    constexpr usearch_sequence_metric_t overlap_k = usearch_sequence_metric_overlap_k;
    constexpr usearch_sequence_metric_t aligned_k = usearch_sequence_metric_needleman_wunsch_k;
    run("test_sequence_vram_collection<bytes_k>", "2048, 100, 128",
        [&] { test_sequence_vram_collection<bytes_k>(stream_key(seed, "short"), executor, 2048, 100, 128); });
    run("test_sequence_vram_collection<bytes_k>", "512, 200, 64",
        [&] { test_sequence_vram_collection<bytes_k>(stream_key(seed, "long"), executor, 512, 200, 64); });
    run("test_sequence_vram_collection<bytes_k>", "256, 1100, 32",
        [&] { test_sequence_vram_collection<bytes_k>(stream_key(seed, "block"), executor, 256, 1100, 32); });
    run("test_sequence_vram_collection<runes_k>", "512, 200, 64",
        [&] { test_sequence_vram_collection<runes_k>(stream_key(seed, "runes"), executor, 512, 200, 64); });
    run("test_sequence_vram_collection<overlap_k>", "512, 200, 64",
        [&] { test_sequence_vram_collection<overlap_k>(stream_key(seed, "overlap"), executor, 512, 200, 64); });
    run("test_sequence_vram_collection<aligned_k>", "512, 200, 64",
        [&] { test_sequence_vram_collection<aligned_k>(stream_key(seed, "aligned"), executor, 512, 200, 64); });
    // Strided alignments past one tile only need to match the exact costs; a host index over them
    // would spend its minutes on diversity pruning.
    run("test_sequence_vram_collection<aligned_k>", "256, 1100, 8", [&] {
        test_sequence_vram_collection<aligned_k>(stream_key(seed, "aligned_stripes"), executor, 256, 1100, 8,
                                                 recall_reference_t::exact_distances_k);
    });
}

#endif // USEARCH_ARCH_CUDA_ || USEARCH_ARCH_ROCM_

} // namespace unum::usearch::test::sequence
