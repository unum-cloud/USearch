/**
 *  @file test/sequence/vram_index.cpp
 *  @author Ash Vardanian
 *  @date October 15, 2026
 *  @brief The sequence VRAM cell: every metric's kernels on an NVIDIA, AMD or Apple GPU beside the
 *      RAM index on the same data, judged by recall against exact neighbors, every reported
 *      distance against the serial kit the RAM suite checks against its oracle.
 *
 *  Sequences are planted templates over the nucleotides with an eighth of their symbols edited, so
 *  neighbors cluster by template while the bytes differ. A containment counts its text's windows, a
 *  warp to an entry, an overlap scores class bytes on both strands, half its queries reverse
 *  complements, and the alignments pack nucleotides five bits apiece, one query a stripe, two and
 *  several, the first two held to the RAM index's recall, so a walk cutting a pair short that would
 *  have made the radius loses neighbors it should keep. Sequences insert in three batches whose
 *  tapes start past their own first byte, as an Arrow slice's do, so the later batches walk graphs
 *  whose reverse links earlier batches merged. A face takes every batch as stored values, a gap for
 *  its sketch, where it has one, the sequence and its closing padding, and links it by the sketches
 *  its device derives or by its canonical score, so each collection's recall also holds the graph
 *  to a RAM graph built under the canonical score. On every vendor, the suite's own kernel holds
 *  the paired alignment scorer to the single one.
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
#include <vector>      // `std::vector` - strings, offsets, matches

#include "usearch/sequence_index.h"
#include "usearch/sequence/ram_index.hpp"
#if USEARCH_ARCH_CUDA_
#include "usearch/sequence/vram_nvidia_metric.cuh"
#include "usearch/basic/vram_nvidia_indexes.hpp"
#endif
#if USEARCH_ARCH_ROCM_
#include "usearch/sequence/vram_amd_metric.hip"
#include "usearch/basic/vram_amd_indexes.hpp"
#endif
#if USEARCH_ARCH_METAL_
#include "usearch/sequence/vram_apple_index.hpp"
#endif

#include "harness.hpp"
#include "classes.hpp"       // `gpu_class_t`, `cuda_class_k` - the class each vendor's cells run
#include "sequence_kits.hpp" // `sequence_vram_kit`, `sequence_nvidia_fatbin`, `sequence_amd_code_object`

#if USEARCH_ARCH_CUDA_

/** The suite's own fatbin, linked in as raw bytes beside its host half, of the kernels
 *  @c test/sequence/vram_kernels_nvidia.cu defines. */
extern "C" util::byte_t const test_vram_nvidia_bundle_start[], test_vram_nvidia_bundle_end[];

#endif
#if USEARCH_ARCH_ROCM_

/** The suite's own code object, linked in as raw bytes beside its host half, holding the kernels
 *  @c test/sequence/vram_kernels_amd.hip defines. */
extern "C" util::byte_t const test_vram_amd_bundle_start[], test_vram_amd_bundle_end[];

#endif
#if USEARCH_ARCH_METAL_

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc23-extensions"
#endif

/** The suite's own MSL, the kernels @c test/sequence/vram_kernels_apple.metal defines, which the
 *  device compiles at first use behind the sources of the kit they score with. */
constexpr char test_vram_apple_source_k[] = {
#embed "vram_kernels_apple.metal" suffix(, 0)
};

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

#endif // USEARCH_ARCH_METAL_

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

/** Packs every string's nucleotides five bits a code, A, C, G and T in that order. */
static void pack_as_codes(std::vector<std::string>& strings) {
    constexpr std::string_view nucleotides_k = "ACGT";
    for (std::string& symbols : strings) {
        std::vector<u8_t> codes;
        for (char const symbol : symbols) codes.push_back(static_cast<u8_t>(nucleotides_k.find(symbol)));
        std::string packed(packed_codes_bytes(codes.size()), '\0');
        pack_codes(codes, bytes_of(packed.data(), packed.size()));
        symbols = std::move(packed);
    }
}

/** The alignment cell's costs over the nucleotides: transitions one, transversions and gaps
 *  two. */
static constexpr alignment_costs_t vram_nucleotide_costs() noexcept {
    alignment_costs_t costs {};
    costs.classes = 4, costs.gap = 2;
    for (std::size_t first = 0; first != 4; ++first)
        for (std::size_t second = 0; second != 4; ++second)
            costs.substitutions[first][second] = first == second ? 0 : (first & 1) == (second & 1) ? 1 : 2;
    return costs;
}

static constexpr alignment_costs_t vram_nucleotide_costs_k = vram_nucleotide_costs();

/** The overlap cell's classes and costs: the nucleotides' own, whose complements
 * the costs record. */
struct dna_costs_t {
    alignment_alphabet_t alphabet {};
    alignment_costs_t costs {};
};
static dna_costs_t vram_dna_costs() {
    std::vector<i8_t> similarities(alignment_byte_alphabet_k * alignment_byte_alphabet_k);
    dna_similarities(byte_similarities_span_t {similarities});
    dna_costs_t folded;
    expect_equal(alignment_alphabet_t::fold(similarities, dna_gap_k, folded.alphabet, folded.costs),
                 status_code_t::success_k);
    expect_equal(folded.alphabet.fold_complement(folded.costs), status_code_t::success_k);
    return folded;
}

/** Turns the nucleotides of @p strings into the form the cell of @p metric_ stores: packed codes,
 *  class bytes or the bytes as they are. */
template <usearch_sequence_metric_t metric_>
static void store_as_cell(std::vector<std::string>& strings) {
    if constexpr (metric_ == usearch_sequence_metric_protein_alignment_k) pack_as_codes(strings);
    else if constexpr (metric_ == usearch_sequence_metric_dna_overlap_k) {
        dna_costs_t const dna = vram_dna_costs();
        for (std::string& symbols : strings) {
            std::string encoded(symbols.size(), '\0');
            dna.alphabet.encode(bytes_of(symbols.data(), symbols.size()), bytes_of(encoded.data(), encoded.size()));
            symbols = std::move(encoded);
        }
    }
}

/** The costs the cell of @p metric_ scores under: none for a containment. */
template <usearch_sequence_metric_t metric_>
static alignment_costs_t vram_cell_costs() {
    if constexpr (metric_ == usearch_sequence_metric_protein_alignment_k) return vram_nucleotide_costs_k;
    else if constexpr (metric_ == usearch_sequence_metric_dna_overlap_k) return vram_dna_costs().costs;
    else return {};
}

/** The tables the kit of the cell of @p metric_ sketches with, of as many columns as its sketches
 *  take bytes, derived as an index derives them; empty for a kit reading none. */
template <usearch_sequence_metric_t metric_, typename index_type_>
struct vram_cell_tables {
    using sketch_tables_t = typename index_type_::sketch_tables_t;
    spectrum_tables<> spectrum;

    explicit vram_cell_tables(std::size_t sketch_bytes) {
        if constexpr (std::is_same_v<sketch_tables_t, spectrum_view_t>)
            expect_equal(spectrum.derive(vram_cell_costs<metric_>(), sketch_bytes), status_code_t::success_k);
    }

    sketch_tables_t view() const noexcept {
        if constexpr (std::is_same_v<sketch_tables_t, spectrum_view_t>) return spectrum.view();
        else return {};
    }
};

/** Gives a face of the cell of @p metric_ the parameters its kit takes, and no sketch tables. */
template <usearch_sequence_metric_t metric_, typename index_type_>
static void configure_cell_parameters(index_type_& index) {
    using canonical_parameters_t = typename index_type_::canonical_parameters_t;
    alignment_costs_t const costs = vram_cell_costs<metric_>();
    if constexpr (std::is_same_v<canonical_parameters_t, window_parameters_t>) {
        std::span<sz_size_t const> const widths = metric_ == usearch_sequence_metric_utf8_containment_k
                                                      ? std::span<sz_size_t const> {containment_window_widths_k}
                                                      : std::span<sz_size_t const> {banded_overlap_window_widths_k};
        auto const [parameters, status] = window_parameters_t::of(widths, costs);
        expect_equal(status, status_code_t::success_k);
        expect_equal(index.canonical_parameters(parameters), status_code_t::success_k);
    }
    else expect_equal(index.canonical_parameters(costs), status_code_t::success_k);
}

/** Hands the face @p index of @p metric_ what its kit scores and sketches with before its first
 *  reservation: the parameters its kit takes, and the cell's sketch tables for sketches of
 *  @p sketch_bytes, the metric's default unless named. */
template <usearch_sequence_metric_t metric_, typename index_type_>
static void configure_cell(index_type_& index, std::size_t sketch_bytes = default_sketch_bytes(metric_)) {
    configure_cell_parameters<metric_>(index);
    vram_cell_tables<metric_, index_type_> const tables {sketch_bytes};
    expect_equal(index.sketch_tables(tables.view(), sketch_bytes), status_code_t::success_k);
}

/** Strings laid out as a device tape stores them, each behind the gap its sketch of
 *  @p sketch_bytes takes and closed to a word, in a tape starting past its own first bytes, as an
 *  Arrow slice's does. */
struct stored_tape_t {
    std::string tape;
    std::vector<std::size_t> offsets;

    explicit stored_tape_t(std::span<std::string const> strings, std::size_t sketch_bytes, std::size_t origin = 0)
        : tape(origin, '\0'), offsets {origin} {
        for (std::string const& sequence : strings) {
            std::size_t const first = tape.size(), bytes = vram_stored_bytes(sequence.size(), sketch_bytes);
            tape.resize(first + bytes);
            sequence.copy(tape.data() + first + sketch_bytes, sequence.size());
            vram_close_stored(bytes_of(tape.data() + first, bytes), sequence.size(), sketch_bytes);
            offsets.push_back(tape.size());
        }
    }

    /** The stored values of the strings in @b [first,bound). */
    tape_view<byte_t const> view(std::size_t first, std::size_t bound) const noexcept {
        return {reinterpret_cast<byte_t const*>(tape.data()), {offsets.data() + first, bound - first + 1}};
    }
    std::size_t bytes() const noexcept { return offsets.back() - offsets.front(); }
};

/** The metric @p metric_ scores with on the CPU under @p capability_ and @p costs, which it
 *  borrows, sized for @p staging candidates. */
template <usearch_sequence_metric_t metric_, sz_capability_t capability_>
static auto make_sequence_metric(std::size_t staging, alignment_costs_t const& costs) {
    if constexpr (metric_ == usearch_sequence_metric_utf8_containment_k) {
        containment_metric<typename containment_ladder<capability_>::kit_t> metric;
        expect_equal(metric.init(staging), status_code_t::success_k);
        return metric;
    }
    else if constexpr (metric_ == usearch_sequence_metric_dna_overlap_k) {
        banded_overlap_metric<typename banded_overlap_ladder<capability_>::kit_t> metric;
        expect_equal(metric.init(staging, costs), status_code_t::success_k);
        return metric;
    }
    else {
        alignment_metric<typename alignment_ladder<capability_>::kit_t> metric;
        expect_equal(metric.init(staging, costs), status_code_t::success_k);
        return metric;
    }
}

#if USEARCH_ARCH_CUDA_

/** What the NVIDIA collections build on: the executor, the face over the kits of the build's NVIDIA
 *  class and the library's fatbins, its storage and the contexts carved from the executor's run. */
struct nvidia_cells_t {
    using executor_t = cuda_executor_t;
    template <usearch_sequence_metric_t metric_, typename node_id_type_>
    using index_t =
        vram_nvidia_index<sequence_vram_kit<metric_, cuda_class_k>, node_id_type_, sequence_nvidia_fatbin<metric_>>;

    static cuda_virtual_memory_allocator<byte_t> storage(executor_t& executor, int device) noexcept {
        return cuda_virtual_memory_allocator<byte_t> {executor.stream(0).handle.api(),
                                                      test::gpu_drivers().extensions.cuda, device};
    }
    static cuda_arena_allocator_t arena(executor_t& executor) noexcept {
        return cuda_arena_allocator_t {executor.stream(0).memory.arena, executor.stream(0)};
    }

    /** The sharded collections' shards, their storage on a device alone, and the runtime opening
     *  the executor. */
    template <typename index_type_, typename build_type_>
    using indexes_t = us::basic::vram_nvidia_indexes<index_type_, util::uuid128_t, build_type_>;
    static cuda_virtual_memory_allocator<byte_t> storage(int device) noexcept {
        return cuda_virtual_memory_allocator<byte_t> {test::gpu_drivers().cuda, test::gpu_drivers().extensions.cuda,
                                                      device};
    }
    static cuda_api_t const& api() noexcept { return test::gpu_drivers().cuda; }
    inline static constexpr gpu_vendor_t vendor_k = gpu_vendor_t::nvidia_k;
    inline static constexpr char runtime_k[] = "CUDA";

    /** The suite's own kernels, the kit whose scorers they pit against each other, and the device
     *  memory they read, launched a warp to a block on the null stream, which each copy drains
     *  before its host bytes go. */
    using paired_kit_t = packed_ampere_alignment_kit_t;
    using module_t = cuda_module_t;
    using device_bytes_t = cuda_unique_ptr<byte_t, cuda_device_allocator<byte_t>>;
    static expected<module_t> suite_kernels(executor_t& executor) noexcept {
        return module_t::load(executor.stream(0).handle.api(), static_cast<int>(executor.stream(0).ordinal),
                              {test_vram_nvidia_bundle_start,
                               static_cast<std::size_t>(test_vram_nvidia_bundle_end - test_vram_nvidia_bundle_start)});
    }
    static device_bytes_t allocate(executor_t& executor, std::size_t bytes) noexcept {
        return cuda_make_unique(cuda_device_allocator<byte_t> {executor.stream(0).handle.api()}, bytes);
    }
    static bool upload(executor_t& executor, device_bytes_t const& to, void const* from, std::size_t bytes) noexcept {
        cuda_api_t const& api = executor.stream(0).handle.api();
        return succeeded(cuda_vendor_t::copy(api, to.get(), from, bytes, {})) && api.stream_synchronize({}) == 0;
    }
    static bool download(executor_t& executor, void* to, device_bytes_t const& from, std::size_t bytes) noexcept {
        cuda_api_t const& api = executor.stream(0).handle.api();
        return succeeded(cuda_vendor_t::copy(api, to, from.get(), bytes, {})) && api.stream_synchronize({}) == 0;
    }
    template <typename... argument_types_>
    static status_t launch(executor_t& executor, module_t::function_t kernel, u32_t blocks,
                           argument_types_ const&... arguments) noexcept {
        return cuda_launch(executor.stream(0).handle.api(), kernel, blocks, paired_kit_t::warp_t::width_k, 0u,
                           cuda_stream_handle_t {}, cuda_overlap_t::serialized_k, argument(arguments)...);
    }

    /** A kernel takes a buffer by its device address and a value as it is. */
    static byte_t* argument(device_bytes_t const& buffer) noexcept { return buffer.get(); }
    static u32_t argument(u32_t value) noexcept { return value; }
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
    using index_t = vram_amd_index<sequence_vram_kit<metric_, gpu_class_t::amd_cdna3_k>, node_id_type_,
                                   sequence_amd_code_object<metric_, gpu_class_t::amd_cdna3_k>>;

    static hip_virtual_memory_allocator<byte_t> storage(executor_t& executor, int device) noexcept {
        return hip_virtual_memory_allocator<byte_t> {executor.stream(0).handle.api(),
                                                     test::gpu_drivers().extensions.hip, device};
    }
    static hip_arena_allocator_t arena(executor_t& executor) noexcept {
        return hip_arena_allocator_t {executor.stream(0).memory.arena, executor.stream(0)};
    }

    /** The sharded collections' shards, their storage on a device alone, and the runtime opening
     *  the executor. */
    template <typename index_type_, typename build_type_>
    using indexes_t = us::basic::vram_amd_indexes<index_type_, util::uuid128_t, build_type_>;
    static hip_virtual_memory_allocator<byte_t> storage(int device) noexcept {
        return hip_virtual_memory_allocator<byte_t> {test::gpu_drivers().hip, test::gpu_drivers().extensions.hip,
                                                     device};
    }
    static hip_api_t const& api() noexcept { return test::gpu_drivers().hip; }
    inline static constexpr gpu_vendor_t vendor_k = gpu_vendor_t::amd_k;
    inline static constexpr char runtime_k[] = "ROCm";

    /** The suite's own kernels, the kit whose scorers they pit against each other, and the device
     *  memory they read, launched a wavefront to a block on the null stream, which each copy drains
     *  before its host bytes go. */
    using paired_kit_t = packed_cdna3_alignment_kit_t;
    using module_t = hip_module_t;
    using device_bytes_t = hip_unique_ptr<byte_t, hip_device_allocator<byte_t>>;
    static expected<module_t> suite_kernels(executor_t& executor) noexcept {
        return module_t::load(executor.stream(0).handle.api(), static_cast<int>(executor.stream(0).ordinal),
                              {test_vram_amd_bundle_start,
                               static_cast<std::size_t>(test_vram_amd_bundle_end - test_vram_amd_bundle_start)});
    }
    static device_bytes_t allocate(executor_t& executor, std::size_t bytes) noexcept {
        return hip_make_unique(hip_device_allocator<byte_t> {executor.stream(0).handle.api()}, bytes);
    }
    static bool upload(executor_t& executor, device_bytes_t const& to, void const* from, std::size_t bytes) noexcept {
        hip_api_t const& api = executor.stream(0).handle.api();
        return succeeded(hip_vendor_t::copy(api, to.get(), from, bytes, {})) && api.stream_synchronize({}) == 0;
    }
    static bool download(executor_t& executor, void* to, device_bytes_t const& from, std::size_t bytes) noexcept {
        hip_api_t const& api = executor.stream(0).handle.api();
        return succeeded(hip_vendor_t::copy(api, to, from.get(), bytes, {})) && api.stream_synchronize({}) == 0;
    }
    template <typename... argument_types_>
    static status_t launch(executor_t& executor, module_t::function_t kernel, u32_t blocks,
                           argument_types_ const&... arguments) noexcept {
        return hip_launch(executor.stream(0).handle.api(), kernel, blocks, paired_kit_t::warp_t::width_k, 0u,
                          hip_stream_handle_t {}, argument(arguments)...);
    }

    /** A kernel takes a buffer by its device address and a value as it is. */
    static byte_t* argument(device_bytes_t const& buffer) noexcept { return buffer.get(); }
    static u32_t argument(u32_t value) noexcept { return value; }
};
using cell_vendor_t = amd_cells_t;

#endif
#if USEARCH_ARCH_METAL_ && !USEARCH_ARCH_CUDA_

/** What the Apple collections build on: the executor, the face over Metal's kits, its storage and
 *  the contexts over its stream. */
struct apple_cells_t {
    using executor_t = metal_executor_t;
    template <usearch_sequence_metric_t metric_, typename node_id_type_>
    using index_t = vram_apple_index<sequence_vram_kit<metric_, gpu_class_t::apple_metal7_k>, node_id_type_>;

    static metal_shared_allocator_t storage(executor_t& executor, int) noexcept {
        return metal_shared_allocator_t {*executor.stream(0).handle.context(),
                                         static_cast<int>(executor.stream(0).ordinal), *executor.stream(0).kernels};
    }
    static metal_shared_allocator_t arena(executor_t& executor) noexcept {
        return metal_shared_allocator_t {executor.stream(0)};
    }

    /** The suite's own kernels, built into the process's pipelines on the stream's device, the kit
     *  whose scorers they pit against each other, and the shared memory they read, launched a
     *  simdgroup to a threadgroup on the stream's queue, which each launch waits for. */
    using paired_kit_t = apple_packed_alignment_kit_t;
    using device_bytes_t = metal_buffer_t;
    struct module_t {
        using function_t = void*;
        executor_t::stream_t* stream = nullptr;

        /** The pipeline of the kernel exported as @p name, or null when the source builds none. */
        function_t function(char const* name) const noexcept {
            metal_context_t const& context = *stream->handle.context();
            auto const [state, status] = stream->kernels->pipeline(*context.api, context.device, source(), name);
            return succeeded(status) ? state.pipeline : nullptr;
        }

        /** The contexts' vocabulary, the kit's pieces and the suite's kernels, spliced once, as its
         *  address keys the library built from it. */
        static char const* source() {
            static std::string const spliced = [] {
                std::string text = std::string {metal_context_source_k} + '\n';
                for (char const* const piece : paired_kit_t::metal_sources_k) text.append(piece).push_back('\n');
                return text.append(test_vram_apple_source_k);
            }();
            return spliced.c_str();
        }
    };
    static expected<module_t> suite_kernels(executor_t& executor) noexcept {
        executor_t::stream_t& stream = executor.stream(0);
        if (!stream.handle.context() || !stream.kernels) return {{}, status_code_t::device_k};
        return {module_t {&stream}, status_code_t::success_k};
    }
    static device_bytes_t allocate(executor_t& executor, std::size_t bytes) noexcept {
        return metal_allocate(*executor.stream(0).handle.context(), bytes);
    }
    static bool upload(executor_t&, device_bytes_t const& to, void const* from, std::size_t bytes) noexcept {
        std::memcpy(to.host, from, bytes);
        return true;
    }
    static bool download(executor_t&, void* to, device_bytes_t const& from, std::size_t bytes) noexcept {
        std::memcpy(to, from.host, bytes);
        return true;
    }
    template <typename... argument_types_>
    static status_t launch(executor_t& executor, module_t::function_t kernel, u32_t blocks,
                           argument_types_ const&... arguments) noexcept {
        executor_t::stream_t& stream = executor.stream(0);
        metal_launch_t commands {*stream.handle.context()->api, stream.handle.queue(), kernel};
        std::size_t slot = 0;
        (bind(commands, arguments, slot++), ...);
        if (status_t const submitted = commands.submit(blocks, metal_simdgroup_width_k); failed(submitted))
            return submitted;
        return commands.wait();
    }

    /** A kernel takes a buffer bound at its slot and a value copied into its slot. */
    static void bind(metal_launch_t& commands, device_bytes_t const& buffer, std::size_t slot) noexcept {
        commands.bind(buffer, slot);
    }
    static void bind(metal_launch_t& commands, u32_t const& value, std::size_t slot) noexcept {
        commands.bind_bytes(&value, sizeof(value), slot);
    }
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
 *      queries, built on the GPU with @p node_id_type_ IDs and in RAM alike; a recall is judged
 *      over @c recall_samples_k of them, and fewer only match the exact distances.
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
    alignment_costs_t const costs = vram_cell_costs<metric_>();

    std::vector<std::string> strings = generate_nucleotides(count + queries_count, length, seed);
    // Every other query of an overlap is a reverse complement, which only the other strand matches.
    if constexpr (metric_ == usearch_sequence_metric_dna_overlap_k)
        for (std::size_t query = count; query < strings.size(); query += 2) {
            std::string& symbols = strings[query];
            std::string const forward = symbols;
            for (std::size_t symbol = 0; symbol != forward.size(); ++symbol)
                symbols[symbol] = dna_complement(forward[forward.size() - 1 - symbol]);
        }
    store_as_cell<metric_>(strings);
    std::string tape(origin_k, '\0');
    std::vector<std::size_t> offsets {origin_k};
    for (std::string const& symbols : strings) tape += symbols, offsets.push_back(tape.size());
    byte_t const* const tape_bytes = reinterpret_cast<byte_t const*>(tape.data());

    // The exact distances of every query to every stored string, through the serial kit.
    nk::matrix<sz_size_t> exact = make_matrix<sz_size_t>(queries_count, count);
    auto serial_metric = make_sequence_metric<metric_, sz_cap_serial_k>(count, costs);
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
        auto metric = make_sequence_metric<metric_, sz_cap_cpus_k>(expansion_k * connectivity_32_k, costs);
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

    // The GPU index, in three batches of stored values, each over its own context.
    int const device = static_cast<int>(executor.stream(0).ordinal);
    vram_index_t vram_index {cell_vendor_t::storage(executor, device)};
    configure_cell<metric_>(vram_index);
    stored_tape_t const stored {std::span<std::string const> {strings}.first(count), default_sketch_bytes(metric_),
                                origin_k};
    expect_equal(vram_index.reserve({count, connectivity_32_k, stored.bytes()}).status, status_code_t::success_k);
    typename vram_index_t::insert_config_t const insert_config {expansion_k};
    nk::vector<util::uuid128_t> keys = make_vector<util::uuid128_t>(count);
    for (std::size_t entry = 0; entry != count; ++entry) keys[entry] = uuid_of(entry);
    for (std::size_t batch = 0; batch != batches_k && held(); ++batch) {
        std::size_t const first = count * batch / batches_k, bound = count * (batch + 1) / batches_k;
        expect_equal(reserve_scratch(executor.stream(0),
                                     vram_index.traversal_memory_usage(stored.view(first, bound), insert_config)),
                     status_code_t::success_k);
        typename vram_index_t::traversal_context_t context {cell_vendor_t::arena(executor)};
        auto const [inserted, status] = vram_index.insert({keys.data() + first, bound - first},
                                                          stored.view(first, bound), context, insert_config);
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
    expect_equal(
        reserve_scratch(executor.stream(0), vram_index.traversal_memory_usage(
                                                tape_view<byte_t const> {tape_bytes, query_offsets}, search_config)),
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
            // The candidates keep 31 bits of a float, so containment's fixed point loses a
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
    if (vram_recall + recall_margin_k < ram_recall)
        fmt::print(stderr, "  recall at ten on {} strings of {} bytes: {:.3f} on the GPU, {:.3f} in RAM\n", count,
                   length, vram_recall, ram_recall);
    expect_true(vram_recall + recall_margin_k >= ram_recall);
    expect_errors("vram sequence distances", comparison_family_t::approximate_k, comparison_verdict_t::assert_k,
                  distance_errors, 0);
}

#endif // USEARCH_ARCH_CUDA_ || USEARCH_ARCH_ROCM_ || USEARCH_ARCH_METAL_

#if USEARCH_ARCH_CUDA_ || USEARCH_ARCH_ROCM_

/** Builds the alignment shards of a sharded index, unreserved, on the device it names, over the
 *  cells of @p cells_type_ 's vendor, which keeps each vendor's builder a symbol of its own. */
template <typename cells_type_>
struct sequence_vram_shards {
    using index_t = typename cells_type_::template index_t<usearch_sequence_metric_protein_alignment_k, u32_t>;

    std::unique_ptr<index_t> operator()(int device) const noexcept {
        std::unique_ptr<index_t> shard(new (std::nothrow) index_t {cells_type_::storage(device)});
        if (shard) configure_cell<usearch_sequence_metric_protein_alignment_k>(*shard);
        return shard;
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
    expect_equal(saved.save(first_path, codec_t {}, executor, config, "protein-alignment"), status_code_t::success_k);
    indexes_type_ loaded {forkunion_allocator_t {no_memory_domain_k}, sequence_vram_shards_t {}};
    loaded.bound_shards(shard_capacity);
    expect_equal(loaded.emplace(static_cast<int>(executor.stream(0).ordinal)), status_code_t::success_k);
    expect_equal(loaded.load(std::string_view {first_path}, codec_t {}, executor, load_config_t {}, connectivity_32_k),
                 status_code_t::success_k);
    expect_equal(loaded.size(), saved.size());
    expect_equal(loaded.save(second_path, codec_t {}, executor, config, "protein-alignment"), status_code_t::success_k);
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
    constexpr std::size_t entries_k = 5000, shard_capacity_k = 1000, wanted_k = 10, expansion_k = 64;

    std::vector<std::string> strings = generate_nucleotides(entries_k + recall_samples_k, 100, seed);
    store_as_cell<usearch_sequence_metric_protein_alignment_k>(strings);
    std::string tape;
    std::vector<std::size_t> offsets {0};
    for (std::string const& symbols : strings) tape += symbols, offsets.push_back(tape.size());
    byte_t const* const tape_bytes = reinterpret_cast<byte_t const*>(tape.data());
    stored_tape_t const stored {std::span<std::string const> {strings}.first(entries_k),
                                default_sketch_bytes(usearch_sequence_metric_protein_alignment_k)};
    tape_view<byte_t const> const sequences = stored.view(0, entries_k);
    tape_view<byte_t const> const queries {tape_bytes, {offsets.data() + entries_k, recall_samples_k + 1}};
    nk::matrix<sz_size_t> exact = make_matrix<sz_size_t>(recall_samples_k, entries_k);
    alignment_costs_t const costs = vram_cell_costs<usearch_sequence_metric_protein_alignment_k>();
    auto serial_metric = make_sequence_metric<usearch_sequence_metric_protein_alignment_k, sz_cap_cpus_k>(entries_k,
                                                                                                          costs);
    for (std::size_t query = 0; query != recall_samples_k; ++query) {
        serial_metric.reset(std::string_view {strings[entries_k + query]});
        for (std::size_t entry = 0; entry != entries_k; ++entry) serial_metric.push(std::string_view {strings[entry]});
        nk::vector_span<sz_size_t> const distances = exact.span().row(query).as_vector();
        serial_metric.drain(to_writable_span(distances), us::basic::unbounded_radius<sz_size_t>());
    }
    nk::vector<util::uuid128_t> keys = make_vector<util::uuid128_t>(entries_k);
    for (std::size_t entry = 0; entry != entries_k; ++entry) keys[entry] = uuid_of(entry);

    indexes_t indexes {forkunion_allocator_t {no_memory_domain_k}, sequence_vram_shards_t {}};
    indexes.bound_shards(shard_capacity_k);
    expect_equal(indexes.emplace(static_cast<int>(executor.stream(0).ordinal)), status_code_t::success_k);
    expect_equal(indexes.emplace(static_cast<int>(executor.stream(0).ordinal)), status_code_t::success_k);
    expect_equal(indexes.reserve({entries_k, connectivity_32_k, stored.bytes()}).status, status_code_t::success_k);
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

    nk::matrix<util::uuid128_t> found_keys = make_matrix<util::uuid128_t>(recall_samples_k, wanted_k);
    nk::matrix<f32_t> found_distances = make_matrix<f32_t>(recall_samples_k, wanted_k);
    nk::vector<std::size_t> counts = make_vector<std::size_t>(recall_samples_k);
    typename index_t::search_config_t const search_config {expansion_k, wanted_k};
    expect_equal(indexes
                     .search(queries, found_keys.span(), found_distances.span(), to_writable_span(counts),
                             search_config, executor)
                     .status,
                 status_code_t::success_k);
    std::size_t sharded_matched = 0;
    for (std::size_t query = 0; query != recall_samples_k; ++query) {
        std::vector<sz_size_t> found;
        for (std::size_t rank = 0; rank != counts[query]; ++rank)
            found.push_back(static_cast<sz_size_t>(found_distances(query, rank)));
        sharded_matched += matched_count(found, exact.view().row(query).as_vector(), wanted_k);
    }

    index_t single {cell_vendor_t::storage(executor, static_cast<int>(executor.stream(0).ordinal))};
    configure_cell<usearch_sequence_metric_protein_alignment_k>(single);
    expect_equal(single.reserve({0, connectivity_32_k, 0}).status, status_code_t::success_k);
    typename index_t::insert_config_t const insert_config {expansion_k};
    expect_equal(reserve_scratch(executor.stream(0), max_of(single.traversal_memory_usage(sequences, insert_config),
                                                            single.traversal_memory_usage(queries, search_config))),
                 status_code_t::success_k);
    typename index_t::traversal_context_t context {cell_vendor_t::arena(executor)};
    expect_equal(single.insert(to_span(keys), sequences, context, insert_config).status, status_code_t::success_k);
    for (util::uuid128_t const& key : keys) miscounted += single.count(key) != 1;
    expect_equal(miscounted, std::size_t {0});
    nk::matrix<u32_t> ids = make_matrix<u32_t>(recall_samples_k, wanted_k);
    expect_equal(single.search(queries, ids.span(), found_distances.span(), search_config, context).status,
                 status_code_t::success_k);
    std::size_t single_matched = 0;
    for (std::size_t query = 0; query != recall_samples_k; ++query) {
        std::vector<sz_size_t> found;
        for (std::size_t rank = 0; rank != wanted_k && ids(query, rank) != (std::numeric_limits<u32_t>::max)(); ++rank)
            found.push_back(static_cast<sz_size_t>(found_distances(query, rank)));
        single_matched += matched_count(found, exact.view().row(query).as_vector(), wanted_k);
    }
    expect_true(sharded_matched + recall_margin_k * recall_samples_k * wanted_k >= single_matched);
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

    std::vector<std::string> strings = generate_nucleotides(entries_k, 100, seed);
    store_as_cell<usearch_sequence_metric_protein_alignment_k>(strings);
    stored_tape_t const stored {strings, default_sketch_bytes(usearch_sequence_metric_protein_alignment_k)};
    tape_view<byte_t const> const sequences = stored.view(0, entries_k);
    std::vector<util::uuid128_t> keys(entries_k);
    for (std::size_t entry = 0; entry != entries_k; ++entry) keys[entry] = uuid_of(entry);

    auto placed = [&]() {
        auto indexes = std::make_unique<indexes_t>(forkunion_allocator_t {no_memory_domain_k},
                                                   sequence_vram_shards_t {});
        expect_equal(indexes->emplace(static_cast<int>(executor.stream(0).ordinal)), status_code_t::success_k);
        expect_equal(indexes->emplace(static_cast<int>(executor.stream(0).ordinal)), status_code_t::success_k);
        return indexes;
    };
    std::unique_ptr<indexes_t> const source = placed(), first = placed(), second = placed(), lone = placed();
    expect_equal(source->reserve({entries_k, connectivity_32_k, stored.bytes()}).status, status_code_t::success_k);
    expect_equal(source->insert(keys, sequences, executor, {expansion_k}).status, status_code_t::success_k);
    auto narrow = std::make_unique<indexes_t>(forkunion_allocator_t {no_memory_domain_k}, sequence_vram_shards_t {});
    expect_equal(narrow->emplace(static_cast<int>(executor.stream(0).ordinal)), status_code_t::success_k);

    // The second source holds keys of its own, so a refusal that moved it would show in its size.
    std::vector<std::string> lone_strings = generate_nucleotides(lone_k, 100, seed + 1);
    store_as_cell<usearch_sequence_metric_protein_alignment_k>(lone_strings);
    stored_tape_t const lone_stored {lone_strings, default_sketch_bytes(usearch_sequence_metric_protein_alignment_k)};
    tape_view<byte_t const> const lone_sequences = lone_stored.view(0, lone_k);
    std::vector<util::uuid128_t> lone_keys(lone_k);
    for (std::size_t entry = 0; entry != lone_k; ++entry) lone_keys[entry] = uuid_of(entries_k + entry);
    expect_equal(lone->reserve({lone_k, connectivity_32_k, lone_stored.bytes()}).status, status_code_t::success_k);
    expect_equal(lone->insert(lone_keys, lone_sequences, executor, {expansion_k}).status, status_code_t::success_k);

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

    std::vector<std::string> strings = generate_nucleotides(entries_k, 100, seed);
    store_as_cell<usearch_sequence_metric_protein_alignment_k>(strings);
    stored_tape_t const stored {strings, default_sketch_bytes(usearch_sequence_metric_protein_alignment_k)};
    tape_view<byte_t const> const sequences = stored.view(0, entries_k);
    std::vector<util::uuid128_t> keys(entries_k);
    for (std::size_t entry = 0; entry != entries_k; ++entry) keys[entry] = uuid_of(entry);

    indexes_t source {forkunion_allocator_t {no_memory_domain_k}, sequence_vram_shards_t {}};
    indexes_t lone {forkunion_allocator_t {no_memory_domain_k}, sequence_vram_shards_t {}};
    for (indexes_t* placed : {&source, &lone}) {
        expect_equal(placed->emplace(static_cast<int>(executor.stream(0).ordinal)), status_code_t::success_k);
        expect_equal(placed->emplace(static_cast<int>(executor.stream(0).ordinal)), status_code_t::success_k);
    }
    expect_equal(source.reserve({entries_k, connectivity_32_k, stored.bytes()}).status, status_code_t::success_k);
    expect_equal(source.insert(keys, sequences, executor, {expansion_k}).status, status_code_t::success_k);

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

    std::vector<std::string> strings = generate_nucleotides(entries_k, 100, seed);
    // Letters already folded, so a text reads back as it was given.
    for (std::string& symbols : strings)
        for (char& symbol : symbols) symbol = static_cast<char>(symbol | 0x20);
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
    init.metric = usearch_sequence_metric_utf8_containment_k;
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

#if USEARCH_ARCH_CUDA_

/** The round trip of @c ram_index.cpp through a checkpoint, on the device the listing names at
 *  @p ordinal under @p residency. */
void test_sequence_sketched_round_trip(u64_t seed, usearch_residency_t residency, std::int32_t ordinal);

/** Adopted rows replace the stored ones on the device: a fixed row drops its sentinel, keeps the
 *  rest in order and pads with sentinels, and a ragged row naming no node is refused. */
static void test_sequence_vram_adopted(u64_t seed, cuda_executor_t& executor) {
    using vram_index_t = nvidia_cells_t::index_t<usearch_sequence_metric_protein_alignment_k, u32_t>;
    constexpr std::size_t entries_k = 64, degree_k = 3, expansion_k = 64;
    constexpr std::size_t connectivity_k = static_cast<std::size_t>(connectivity_32_k);
    constexpr u32_t sentinel_k = (std::numeric_limits<u32_t>::max)();
    std::vector<std::string> strings = generate_nucleotides(entries_k, 100, seed);
    store_as_cell<usearch_sequence_metric_protein_alignment_k>(strings);
    stored_tape_t const stored {strings, default_sketch_bytes(usearch_sequence_metric_protein_alignment_k)};
    tape_view<byte_t const> const sequences = stored.view(0, entries_k);
    std::vector<util::uuid128_t> keys(entries_k);
    for (std::size_t entry = 0; entry != entries_k; ++entry) keys[entry] = uuid_of(entry);

    vram_index_t index {nvidia_cells_t::storage(executor, static_cast<int>(executor.stream(0).ordinal))};
    configure_cell<usearch_sequence_metric_protein_alignment_k>(index);
    expect_equal(index.reserve({entries_k, connectivity_32_k, stored.bytes()}).status, status_code_t::success_k);
    vram_index_t::insert_config_t const insert_config {expansion_k};
    expect_equal(reserve_scratch(executor.stream(0), index.traversal_memory_usage(sequences, insert_config)),
                 status_code_t::success_k);
    vram_index_t::traversal_context_t context {nvidia_cells_t::arena(executor)};
    expect_equal(index.insert(keys, sequences, context, insert_config).status, status_code_t::success_k);

    cuda_api_t const& api = executor.stream(0).handle.api();
    auto const uploaded = [&]<typename element_type_>(std::vector<element_type_> const& host) {
        cuda_unique_ptr<element_type_> device = cuda_make_unique(cuda_device_allocator<element_type_> {api},
                                                                 host.size());
        expect_true(device &&
                    cuda_copy(api, device.get(), host.data(), host.size() * sizeof(element_type_)) == cuda_success_k);
        return device;
    };
    std::vector<u32_t> fixed(entries_k * degree_k);
    for (std::size_t node = 0; node != entries_k; ++node) {
        fixed[node * degree_k] = static_cast<u32_t>((node + 1) % entries_k);
        fixed[node * degree_k + 1] = sentinel_k;
        fixed[node * degree_k + 2] = static_cast<u32_t>((node + 2) % entries_k);
    }
    cuda_unique_ptr<u32_t> const device_fixed = uploaded(fixed);
    expect_equal(index.adopt_links({{device_fixed.get(), fixed.size()}, {}, degree_k}, context),
                 status_code_t::success_k);
    std::vector<u32_t> rows(entries_k * connectivity_k);
    expect_equal(index.neighbors(0, rows), status_code_t::success_k);
    for (std::size_t node = 0; node != entries_k; ++node) {
        u32_t const* const row = rows.data() + node * connectivity_k;
        expect_equal(row[0], static_cast<u32_t>((node + 1) % entries_k));
        expect_equal(row[1], static_cast<u32_t>((node + 2) % entries_k));
        for (std::size_t slot = 2; slot != connectivity_k; ++slot) expect_equal(row[slot], sentinel_k);
    }

    std::vector<u64_t> ragged_offsets(entries_k + 1, 0);
    ragged_offsets.back() = 1;
    std::vector<u32_t> const ragged {static_cast<u32_t>(entries_k)};
    cuda_unique_ptr<u64_t> const device_offsets = uploaded(ragged_offsets);
    cuda_unique_ptr<u32_t> const device_ragged = uploaded(ragged);
    expect_equal(index.adopt_links({{device_ragged.get(), 1}, {device_offsets.get(), entries_k + 1}, 0}, context),
                 status_code_t::malformed_k);
}

/** Every value a face stores on the device is its sketch of @p sketched bytes, where the kit has
 *  one, its sequence and its closing padding, read back off the tape: the sketch the device derived
 *  for an inserted batch equals the serial kit's of the same sequence, the sequence sits behind it
 *  byte for byte, the padding is zeroes closing with its own count, and the values a load stores
 *  from a checkpoint read alike. */
template <usearch_sequence_metric_t metric_>
static void test_sequence_vram_sketches(u64_t seed, std::size_t sketched, cuda_executor_t& executor) {
    using vram_index_t = nvidia_cells_t::index_t<metric_, u32_t>;
    using host_kit_t = sequence_kit<metric_, sz_cap_serial_k>;
    constexpr std::size_t entries_k = 96, expansion_k = 64;
    checks_held_t const held;
    // Lengths from none to past two hundred symbols, so every closing count occurs.
    std::vector<std::string> strings;
    for (std::size_t entry = 0; entry != entries_k; ++entry)
        strings.push_back(generate_nucleotides(1, 3 * entry, seed + entry).front());
    store_as_cell<metric_>(strings);
    stored_tape_t const stored {strings, sketched};
    std::vector<util::uuid128_t> keys(entries_k);
    for (std::size_t entry = 0; entry != entries_k; ++entry) keys[entry] = uuid_of(entry);
    vram_cell_tables<metric_, vram_index_t> const tables {sketched};
    cuda_api_t const& api = executor.stream(0).handle.api();

    auto const expect_stored = [&](vram_index_t& index) {
        us::basic::vram_ragged_storage<u32_t> const device = index.storage();
        std::vector<u64_t> starts(entries_k + 1);
        if (!expect_true(cuda_copy(api, starts.data(), device.offsets, starts.size() * sizeof(u64_t)) ==
                         cuda_success_k))
            return;
        std::vector<byte_t> values(starts.back());
        if (!expect_true(cuda_copy(api, values.data(), device.values, values.size()) == cuda_success_k)) return;
        std::size_t mismatched = 0;
        for (std::size_t entry = 0; entry != entries_k; ++entry) {
            bytes_view_t const value {values.data() + starts[entry], starts[entry + 1] - starts[entry]};
            std::string const& sequence = strings[entry];
            std::size_t const closing = value.size() - sketched - sequence.size();
            bool laid = starts[entry] % 4 == 0 && value.size() == vram_stored_bytes(sequence.size(), sketched) &&
                        std::to_integer<std::size_t>(value.back()) == closing &&
                        std::memcmp(value.data() + sketched, sequence.data(), sequence.size()) == 0;
            for (std::size_t pad = 0; laid && pad + 1 < closing; ++pad)
                laid = value[sketched + sequence.size() + pad] == byte_t {0};
            mismatched += !laid;
            if constexpr (a_sketching_kit<host_kit_t>) {
                std::vector<byte_t> serial(sketched), scratch(host_kit_t::sketch_scratch_bytes(sketched));
                host_kit_t::sketch_write(tables.view(), bytes_of(sequence.data(), sequence.size()), serial, scratch);
                mismatched += laid && std::memcmp(value.data(), serial.data(), sketched) != 0;
            }
        }
        expect_equal(mismatched, std::size_t {0});
    };

    // A smooth sketch's face that kept no tables refuses its first reservation.
    if constexpr (!std::is_empty_v<typename vram_index_t::sketch_tables_t>) {
        vram_index_t untabled {nvidia_cells_t::storage(executor, static_cast<int>(executor.stream(0).ordinal))};
        configure_cell_parameters<metric_>(untabled);
        expect_equal(untabled.reserve({entries_k, connectivity_32_k, stored.bytes()}).status,
                     status_code_t::uninitialized_k);
    }

    // Sketch ranks a block cannot hold in shared memory are refused at the first reservation.
    if constexpr (std::is_empty_v<typename vram_index_t::sketch_tables_t> && a_sketching_kit<host_kit_t>) {
        auto const [specs, query_status] = cuda_device_specs_t::query(api,
                                                                      static_cast<int>(executor.stream(0).ordinal));
        expect_equal(query_status, status_code_t::success_k);
        vram_index_t oversketched {nvidia_cells_t::storage(executor, static_cast<int>(executor.stream(0).ordinal))};
        configure_cell<metric_>(oversketched,
                                round_up_to_multiple<std::size_t>(specs.shared_memory_per_block_bytes, 4));
        expect_equal(oversketched.reserve({entries_k, connectivity_32_k, stored.bytes()}).status,
                     status_code_t::config_mismatch_k);
    }

    vram_index_t index {nvidia_cells_t::storage(executor, static_cast<int>(executor.stream(0).ordinal))};
    configure_cell<metric_>(index, sketched);
    expect_equal(index.reserve({entries_k, connectivity_32_k, stored.bytes()}).status, status_code_t::success_k);
    typename vram_index_t::insert_config_t const insert_config {expansion_k};
    expect_equal(
        reserve_scratch(executor.stream(0), index.traversal_memory_usage(stored.view(0, entries_k), insert_config)),
        status_code_t::success_k);
    typename vram_index_t::traversal_context_t context {nvidia_cells_t::arena(executor)};

    // A tape of bare sequences is refused whole, and claims nothing.
    std::string bare_bytes;
    std::vector<std::size_t> bare_offsets {0};
    for (std::string const& sequence : strings) bare_bytes += sequence, bare_offsets.push_back(bare_bytes.size());
    tape_view<byte_t const> const bare {reinterpret_cast<byte_t const*>(bare_bytes.data()), bare_offsets};
    expect_equal(index.insert(keys, bare, context, insert_config).status, status_code_t::config_mismatch_k);
    expect_equal(index.size(), std::size_t {0});

    expect_equal(index.insert(keys, stored.view(0, entries_k), context, insert_config).status,
                 status_code_t::success_k);
    if (held()) expect_stored(index);

    // A checkpoint holds the sequences alone, and a load lays every value out again.
    auto const [extents, extents_status] = index.columns_extents(context);
    expect_equal(extents_status, status_code_t::success_k);
    std::size_t sequences_bytes = 0;
    for (std::string const& sequence : strings) sequences_bytes += sequence.size();
    expect_equal(extents.value_bytes, sequences_bytes);
    std::vector<util::uuid128_t> saved_keys(extents.nodes);
    std::vector<u64_t> saved_neighbor_offsets(extents.nodes + 1), saved_value_offsets(extents.nodes + 1);
    std::vector<u32_t> saved_neighbors(extents.neighbors);
    std::vector<byte_t> saved_values(extents.value_bytes);
    expect_equal(
        index.save({saved_keys, saved_neighbor_offsets, saved_neighbors, saved_value_offsets, saved_values}, context),
        status_code_t::success_k);
    safegraphs_saved_columns<u32_t> columns;
    columns.keys = std::as_bytes(std::span<util::uuid128_t const> {saved_keys});
    columns.key_dtype = dtype_t::uuid_k;
    columns.neighbor_ids = saved_neighbors, columns.neighbor_offsets = saved_neighbor_offsets;
    columns.value_offsets = saved_value_offsets, columns.values = saved_values;
    columns.element_bytes = 1;
    vram_index_t loaded {nvidia_cells_t::storage(executor, static_cast<int>(executor.stream(0).ordinal))};
    configure_cell<metric_>(loaded, sketched);
    expect_equal(loaded.reserve({entries_k, connectivity_32_k, stored.bytes()}).status, status_code_t::success_k);
    expect_equal(loaded.load(columns, context), status_code_t::success_k);
    if (held()) expect_stored(loaded);
}

#endif
#endif // USEARCH_ARCH_CUDA_ || USEARCH_ARCH_ROCM_
#if USEARCH_ARCH_CUDA_ || USEARCH_ARCH_ROCM_ || USEARCH_ARCH_METAL_

/**
 *  @brief The paired alignment scorer answers bit for bit what the single scorer answers for
 *      each of its two candidates.
 *
 *  Queries and candidates span 16 to 1100 protein codes, so a query takes one stripe or several and
 *  a pair's candidates differ in length; a candidate is an edit of its query, an unrelated draw, or
 *  the query itself. Costs are BLOSUM62's folded, under gaps of 4 to 15, and each candidate's
 *  radius is drawn apart: its exact distance, one below, half of it, or none.
 */
static void test_alignment_pairs(u64_t seed, cell_vendor_t::executor_t& executor) {
    using device_bytes_t = cell_vendor_t::device_bytes_t;
    constexpr std::size_t tasks_k = 256, shortest_k = 16, longest_k = 1100;
    constexpr u8_t gaps_k[] = {4, 7, 11, 15};
    checks_held_t const held;
    auto [module, load_status] = cell_vendor_t::suite_kernels(executor);
    if (!expect_true(succeeded(load_status), "The suite's kernels do not load")) return;
    cell_vendor_t::module_t::function_t const kernel = module.function("alignment_pairs_kernel");
    if (!expect_true(kernel != nullptr, "The suite's kernels lack the pairs one")) return;
    std::vector<i8_t> similarities(alignment_byte_alphabet_k * alignment_byte_alphabet_k);
    protein_similarities(byte_similarities_span_t {similarities});
    alignment_alphabet_t alphabet {};
    alignment_costs_t costs {};
    if (!expect_equal(alignment_alphabet_t::fold(similarities, protein_gap_k, alphabet, costs),
                      status_code_t::success_k))
        return;

    split_mix_generator_t random_generator {seed};
    auto const drawn = [&] {
        std::vector<u8_t> codes(shortest_k + random_generator.below(longest_k - shortest_k + 1));
        for (u8_t& code : codes) code = static_cast<u8_t>(random_generator.below(costs.classes));
        return codes;
    };
    // An eighth of the codes substituted, inserted or erased, within the lengths drawn.
    auto const edited = [&](std::vector<u8_t> codes) {
        std::size_t const edits = codes.size() / 8;
        for (std::size_t edit = 0; edit != edits; ++edit) {
            std::size_t const position = random_generator.below(codes.size());
            u8_t const code = static_cast<u8_t>(random_generator.below(costs.classes));
            switch (random_generator.below(3)) {
            case 0: codes[position] = code; break;
            case 1:
                if (codes.size() < longest_k) codes.insert(codes.begin() + position, code);
                break;
            default:
                if (codes.size() > shortest_k) codes.erase(codes.begin() + position);
                break;
            }
        }
        return codes;
    };
    std::vector<std::vector<u8_t>> sequences;
    for (std::size_t task = 0; task != tasks_k; ++task) {
        std::vector<u8_t> const query = drawn();
        sequences.push_back(query);
        for (std::size_t candidate = 0; candidate != 2; ++candidate) switch (random_generator.below(3)) {
            case 0: sequences.push_back(edited(query)); break;
            case 1: sequences.push_back(drawn()); break;
            default: sequences.push_back(query); break;
            }
    }
    std::string tape;
    std::vector<u32_t> offsets {0};
    for (std::vector<u8_t> const& codes : sequences) {
        std::string packed(packed_codes_bytes(codes.size()), '\0');
        pack_codes(codes, bytes_of(packed.data(), packed.size()));
        tape += packed, offsets.push_back(static_cast<u32_t>(tape.size()));
    }

    auto const uploaded = [&](void const* data, std::size_t bytes) {
        device_bytes_t device = cell_vendor_t::allocate(executor, bytes);
        expect_true(device && cell_vendor_t::upload(executor, device, data, bytes));
        return device;
    };
    // Every query's span is as wide as a pack of the longest query's codes prepares into.
    std::string const longest_pack(packed_codes_bytes(longest_k), '\0');
    u32_t const span_bytes = static_cast<u32_t>(cell_vendor_t::paired_kit_t::canonical_span_bytes(
        reinterpret_cast<byte_t const*>(longest_pack.data()), longest_pack.size(), costs));
    device_bytes_t const device_tape = uploaded(tape.data(), tape.size());
    device_bytes_t const device_offsets = uploaded(offsets.data(), offsets.size() * sizeof(u32_t));
    device_bytes_t const spans = cell_vendor_t::allocate(executor, tasks_k * span_bytes);
    device_bytes_t const device_distances = cell_vendor_t::allocate(executor, 4 * tasks_k * sizeof(distance_t));
    if (!expect_true(spans && device_distances) || !held()) return;

    std::vector<u32_t> column(longest_k + 1);
    for (u8_t const gap : gaps_k) {
        costs.gap = gap;
        std::vector<sz_size_t> exact(3 * tasks_k);
        std::vector<distance_t> radii(2 * tasks_k);
        for (std::size_t task = 0; task != tasks_k; ++task)
            for (std::size_t candidate = 1; candidate != 3; ++candidate) {
                std::size_t const slot = 3 * task + candidate;
                exact[slot] = alignment_serial(
                    costs, sequences[3 * task],
                    bytes_of(tape.data() + offsets[slot], std::size_t {offsets[slot + 1] - offsets[slot]}),
                    (std::numeric_limits<sz_size_t>::max)(), column.data());
                distance_t const distance = static_cast<distance_t>(exact[slot]);
                switch (random_generator.below(4)) {
                case 0: radii[2 * task + candidate - 1] = distance; break;
                case 1: radii[2 * task + candidate - 1] = distance - 1; break;
                case 2: radii[2 * task + candidate - 1] = static_cast<distance_t>(exact[slot] / 2); break;
                default: radii[2 * task + candidate - 1] = std::numeric_limits<distance_t>::infinity(); break;
                }
            }
        device_bytes_t const device_costs = uploaded(&costs, sizeof(costs));
        device_bytes_t const device_radii = uploaded(radii.data(), radii.size() * sizeof(distance_t));
        std::vector<distance_t> distances(4 * tasks_k);
        expect_equal(cell_vendor_t::launch(executor, kernel, static_cast<u32_t>(tasks_k), device_costs, device_tape,
                                           device_offsets, device_radii, spans, span_bytes, device_distances),
                     status_code_t::success_k);
        expect_true(cell_vendor_t::download(executor, distances.data(), device_distances,
                                            distances.size() * sizeof(distance_t)));
        if (!held()) return;
        for (std::size_t task = 0; task != tasks_k; ++task)
            for (std::size_t candidate = 0; candidate != 2; ++candidate) {
                std::size_t const slot = 3 * task + candidate + 1;
                distance_t const single = distances[4 * task + candidate], paired = distances[4 * task + candidate + 2];
                distance_t const radius = radii[2 * task + candidate];
                std::size_t const rows = sequences[3 * task].size(), columns = sequences[slot].size();
                expect_equal(paired, single, "gap {}, task {}, candidate {}: {} by {} codes, exact {}, radius {}", gap,
                             task, candidate, rows, columns, exact[slot], radius);
                if (static_cast<distance_t>(exact[slot]) <= radius)
                    expect_equal(single, static_cast<distance_t>(exact[slot]),
                                 "gap {}, task {}, candidate {}: {} by {} codes, exact {}, radius {}", gap, task,
                                 candidate, rows, columns, exact[slot], radius);
            }
    }
}

#endif
#if USEARCH_ARCH_CUDA_ || USEARCH_ARCH_ROCM_

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
    auto [executor,
          spawn_status] = spawn_harness_executor<cell_vendor_t::executor_t>(std::span<int const> {&ordinal, 1});
    if (!expect_true(succeeded(spawn_status), "{} device #{} opens no stream", cell_vendor_t::runtime_k, ordinal))
        return;
    u64_t const seed = test_run.settings.seed;
    auto const run = [&](char const* call, std::string_view arguments, auto&& check) {
        if (arguments.empty()) run_test_case(fmt::format("{}({})", call, vendor).c_str(), check);
        else run_test_case(fmt::format("{}({}, {})", call, vendor, arguments).c_str(), check);
    };
    constexpr usearch_sequence_metric_t aligned_k = usearch_sequence_metric_protein_alignment_k;
    constexpr usearch_sequence_metric_t overlapped_k = usearch_sequence_metric_dna_overlap_k;
    constexpr usearch_sequence_metric_t containment_k = usearch_sequence_metric_utf8_containment_k;
    run("test_sequence_vram_collection<containment_k>", "2048, 100, recall_samples_k", [&] {
        test_sequence_vram_collection<containment_k>(stream_key(seed, "short"), executor, 2048, 100, recall_samples_k);
    });
    run("test_sequence_vram_collection<containment_k>", "256, 1100, recall_samples_k", [&] {
        test_sequence_vram_collection<containment_k>(stream_key(seed, "block"), executor, 256, 1100, recall_samples_k);
    });
    run("test_sequence_vram_collection<containment_k>", "512, 200, recall_samples_k", [&] {
        test_sequence_vram_collection<containment_k>(stream_key(seed, "overlap"), executor, 512, 200, recall_samples_k);
    });
    run("test_sequence_vram_collection<overlapped_k>", "512, 200, recall_samples_k", [&] {
        test_sequence_vram_collection<overlapped_k>(stream_key(seed, "banded_overlap"), executor, 512, 200,
                                                    recall_samples_k);
    });
    run("test_sequence_vram_collection<aligned_k>", "512, 200, recall_samples_k", [&] {
        test_sequence_vram_collection<aligned_k>(stream_key(seed, "aligned"), executor, 512, 200, recall_samples_k);
    });
    run("test_sequence_vram_collection<aligned_k>", "256, 300, recall_samples_k", [&] {
        test_sequence_vram_collection<aligned_k>(stream_key(seed, "aligned_two_stripes"), executor, 256, 300,
                                                 recall_samples_k);
    });
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
    run("test_alignment_pairs", "",
        [&] { test_alignment_pairs(stream_key(seed, "needleman_wunsch_pairs"), executor); });
#endif
#if USEARCH_ARCH_CUDA_
    run("test_sequence_vram_adopted", "", [&] { test_sequence_vram_adopted(stream_key(seed, "adopted"), executor); });
    // A part of one pass, a pass and a tail, and two passes of a warp's sketching lanes.
    for (std::size_t const sketched : {std::size_t {64}, std::size_t {516}, std::size_t {1024}}) {
        run("test_sequence_vram_sketches<containment_k>", fmt::format("{}", sketched),
            [&] { test_sequence_vram_sketches<containment_k>(stream_key(seed, "hash_sketches"), sketched, executor); });
        run("test_sequence_vram_sketches<aligned_k>", fmt::format("{}", sketched),
            [&] { test_sequence_vram_sketches<aligned_k>(stream_key(seed, "smooth_sketches"), sketched, executor); });
    }
    run("test_sequence_vram_sketches<overlapped_k>", "0",
        [&] { test_sequence_vram_sketches<overlapped_k>(stream_key(seed, "no_sketches"), 0, executor); });
    run("test_sequence_sketched_round_trip", "usearch_residency_vram_k", [&] {
        test_sequence_sketched_round_trip(stream_key(seed, "sketched_round_trip"), usearch_residency_vram_k,
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
    auto [executor, spawn_status] = spawn_harness_executor<metal_executor_t>(std::span<int const> {&ordinal, 1});
    if (!expect_true(succeeded(spawn_status), "Metal device #{} opens no stream", ordinal)) return;
    u64_t const seed = test_run.settings.seed;
    char const* const vendor = constant_of(gpu_vendor_t::apple_k);
    auto const run = [&](char const* call, std::string_view arguments, auto&& check) {
        if (arguments.empty()) run_test_case(fmt::format("{}({})", call, vendor).c_str(), check);
        else run_test_case(fmt::format("{}({}, {})", call, vendor, arguments).c_str(), check);
    };
    constexpr usearch_sequence_metric_t aligned_k = usearch_sequence_metric_protein_alignment_k;
    constexpr usearch_sequence_metric_t overlapped_k = usearch_sequence_metric_dna_overlap_k;
    constexpr usearch_sequence_metric_t containment_k = usearch_sequence_metric_utf8_containment_k;
    run("test_sequence_vram_collection<containment_k>", "2048, 100, recall_samples_k", [&] {
        test_sequence_vram_collection<containment_k>(stream_key(seed, "short"), executor, 2048, 100, recall_samples_k);
    });
    run("test_sequence_vram_collection<containment_k>", "256, 1100, recall_samples_k", [&] {
        test_sequence_vram_collection<containment_k>(stream_key(seed, "block"), executor, 256, 1100, recall_samples_k);
    });
    run("test_sequence_vram_collection<containment_k>", "512, 200, recall_samples_k", [&] {
        test_sequence_vram_collection<containment_k>(stream_key(seed, "overlap"), executor, 512, 200, recall_samples_k);
    });
    run("test_sequence_vram_collection<overlapped_k>", "512, 200, recall_samples_k", [&] {
        test_sequence_vram_collection<overlapped_k>(stream_key(seed, "banded_overlap"), executor, 512, 200,
                                                    recall_samples_k);
    });
    run("test_sequence_vram_collection<aligned_k>", "512, 200, recall_samples_k", [&] {
        test_sequence_vram_collection<aligned_k>(stream_key(seed, "aligned"), executor, 512, 200, recall_samples_k);
    });
    run("test_sequence_vram_collection<aligned_k>", "256, 300, recall_samples_k", [&] {
        test_sequence_vram_collection<aligned_k>(stream_key(seed, "aligned_two_stripes"), executor, 256, 300,
                                                 recall_samples_k);
    });
    // Strided alignments past one tile only need to match the exact costs; a host index over them
    // would spend its minutes on diversity pruning.
    run("test_sequence_vram_collection<aligned_k>", "256, 1100, 8", [&] {
        test_sequence_vram_collection<aligned_k>(stream_key(seed, "aligned_stripes"), executor, 256, 1100, 8,
                                                 recall_reference_t::exact_distances_k);
    });
    run("test_alignment_pairs", "",
        [&] { test_alignment_pairs(stream_key(seed, "needleman_wunsch_pairs"), executor); });
}

#endif // USEARCH_ARCH_CUDA_ || USEARCH_ARCH_ROCM_

} // namespace unum::usearch::test::sequence
