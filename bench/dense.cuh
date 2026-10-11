/**
 *  @file bench/dense.cuh
 *  @author Ash Vardanian
 *  @date July 22, 2026
 *  @brief The backend-neutral benchmarking surface shared by every dense-index residency - volatile
 *      in-RAM, GPU-resident, and persisted alike.
 *
 *  Only what every binary shares lives here: the CLI option set, the RetriEval checkpoint naming
 *  and layout, the dataset streaming trampoline, and the process setup of the CPU binaries. The
 *  search telemetry and judgment live in @c dense_search.cuh, the algebra surface in
 *  @c dense_algebra.cuh. The run loops are backend property, so this header names no index and
 *  reads no index-specific members - the same surface compiles under host compilers and nvcc.
 */
#pragma once

#include <cmath>   // `std::abs` for the rescale amplitude
#include <cstdio>  // `stdout` for the TTY probe
#include <cstdlib> // `std::exit`
#include <cstring> // `std::memcpy` - the checkpoint runs

#include <algorithm>   // `std::ranges::find`, `std::copy_n`, `std::max`
#include <charconv>    // `std::from_chars` - the `--memory-domain` ordinal
#include <ranges>      // `std::views::iota`, `std::views::filter`, `std::views::transform`
#include <span>        // `std::span`
#include <string>      // `std::string`
#include <string_view> // `std::string_view`
#include <type_traits> // `std::integral_constant` for the dispatch chains
#include <vector>      // `std::vector`

#include <fmt/format.h>
#include <fmt/ranges.h> // `fmt::join`

#include <forkunion.h>
#include <forkunion/capabilities.hpp>
#include <numkong/numkong.h>
#include <numkong/numkong.hpp>
#include <stringzilla/stringzilla.hpp>

#include "usearch/utilities/safegraphs_format.hpp"

#if !USEARCH_OS_WINDOWS_
#include <unistd.h> // `isatty`
#endif

#include "arguments.hpp" // `flag_t`, `count_t`, `counts_t`, `switch_t`, `parse_or_exit`
#include "bench.hpp"
#include "utilities/persisted_dataset.hpp" // `mapped_file_t`, `parse_matrix_header`, `matrix_save`

namespace unum::usearch::bench {

#pragma region Options and Styling

/** One named capability ceiling - the row shape the compute and schedule tables share. */
template <typename capability_type_>
struct capability_choice {
    char const* name;
    capability_type_ bit;
};

/** Compute-capability ceilings the kernel chains adapt, ascending per architecture and named by
 *  NumKong's microarchitecture vocabulary; the floor the table never lists is the serial
 *  kernels. */
inline constexpr capability_choice<nk_capability_t> compute_choices_k[] = {
#if NUMKONG_ARCH_X8664_
    {"haswell", nk_cap_haswell_k}, {"skylake", nk_cap_skylake_k},         {"icelake", nk_cap_icelake_k},
    {"genoa", nk_cap_genoa_k},     {"sapphireamx", nk_cap_sapphireamx_k},
#elif NUMKONG_ARCH_ARM64_
    {"neon", nk_cap_neon_k},
    {"neonsdot", nk_cap_neonsdot_k},
    {"neonbfdot", nk_cap_neonbfdot_k},
#else
    {"serial", nk_cap_serial_k},
#endif
};

/** Scheduling ceilings the harness instantiates waiters for, ascending per architecture and named
 *  by ForkUnion's instruction vocabulary; the floor is the STL yield. Every row is gated by the
 *  runtime capability mask at dispatch, since Arm's @c WFET has no compile-time feature macro. */
inline constexpr capability_choice<fu_capabilities_t> schedule_choices_k[] = {
#if FORKUNION_ARCH_X86_64_
    {"pause", fu_capability_x86_pause_k},
    {"tpause", fu_capability_x86_tpause_k},
#elif FORKUNION_ARCH_ARM64_
    {"yield", fu_capability_arm64_yield_k},
    {"wfet", fu_capability_arm64_wfet_k},
#elif FORKUNION_ARCH_RISCV64_
    {"pause", fu_capability_risc5_pause_k},
    {"wrs", fu_capability_risc5_wrs_k},
#else
    {"standard", fu_capabilities_unknown_k},
#endif
};

/** Everything at or below the @p index-th ceiling of @p choices, as one allow-mask. */
template <typename capability_type_, std::size_t count_>
constexpr capability_type_ prefix_mask(capability_choice<capability_type_> const (&choices)[count_],
                                       std::size_t index) noexcept {
    capability_type_ mask {};
    for (std::size_t choice = 0; choice <= index && choice < count_; ++choice)
        mask = static_cast<capability_type_>(mask | choices[choice].bit);
    return mask;
}

/** The @p floor name, then every choice of @p choices whose bit @p mask covers - a view the
 *  report joins in one print. */
template <typename capability_type_, std::size_t count_>
auto named_choices(char const* floor, capability_choice<capability_type_> const (&choices)[count_],
                   capability_type_ mask) noexcept {
    return std::views::iota(std::size_t {0}, count_ + 1) |
           std::views::filter([&choices, mask](std::size_t position) noexcept {
               return position == 0 || (choices[position - 1].bit & mask) != 0;
           }) |
           std::views::transform([floor, &choices](std::size_t position) noexcept {
               return position ? choices[position - 1].name : floor;
           });
}

/** Names a single capability bit of @p choices; the unlisted @p floor backs them all. */
template <typename capability_type_, std::size_t count_>
constexpr char const* bit_name(capability_choice<capability_type_> const (&choices)[count_], capability_type_ bit,
                               char const* floor) noexcept {
    for (capability_choice<capability_type_> const& choice : choices)
        if (choice.bit == bit) return choice.name;
    return floor;
}

/** The index of the newest table row of @p choices inside @p mask, or @c count_ when the
 *  mask names none - the dispatch cascade's entry point into a table. */
template <typename capability_type_, std::size_t count_>
constexpr std::size_t highest_index(capability_choice<capability_type_> const (&choices)[count_],
                                    capability_type_ mask) noexcept {
    std::size_t highest = count_;
    for (std::size_t choice = 0; choice != count_; ++choice)
        if (mask & choices[choice].bit) highest = choice;
    return highest;
}

/** The index of the @p choices row named @p name, or @c count_ when unmatched - the CLI's
 *  entry point into a table, the naming inverse of @ref highest_index. */
template <typename capability_type_, std::size_t count_>
inline std::size_t index_of_name(capability_choice<capability_type_> const (&choices)[count_],
                                 std::string_view name) noexcept {
    for (std::size_t choice = 0; choice != count_; ++choice)
        if (name == choices[choice].name) return choice;
    return count_;
}

/** The schedule allow-mask intersected with what this machine offers - the one place the
 *  @c fu_capabilities_t C-enum arithmetic is re-typed, so its callers never re-cast. */
inline fu_capabilities_t effective_schedule(fu_capabilities_t allow) noexcept {
    return static_cast<fu_capabilities_t>(allow & fu_runtime_capabilities());
}

/** Every schedule bit except the waiter hints - the floor a `--max-schedule` ceiling builds on,
 *  keeping the pool-management toolbox while capping the wait instruction. */
inline constexpr fu_capabilities_t schedule_floor_k = static_cast<fu_capabilities_t>(fu_capabilities_all_k &
                                                                                     ~fu_capability_any_yield_k);

/** Lowers a runtime table position in `[0, count_)` to a compile-time index and hands it to
 *  @p on_row as an @c integral_constant; a position at or past the tail takes @p on_floor. Every
 *  position instantiates its own @p on_row, so the whole grid is built at compile time. */
template <std::size_t count_, std::size_t choice_ = 0>
int dispatch_index(std::size_t position, auto on_row, auto on_floor) {
    if constexpr (choice_ < count_) {
        if (position == choice_) return on_row(std::integral_constant<std::size_t, choice_> {});
        return dispatch_index<count_, choice_ + 1>(position, on_row, on_floor);
    }
    else return on_floor();
}

/** Matches a runtime enum against a compile-time value set, handing the match to @p on_value as an
 *  @c integral_constant; an unlisted value goes to @p on_unknown as the raw runtime value, so its
 *  diagnostic can name what was rejected. */
template <auto value_, auto... rest_>
int dispatch_value(decltype(value_) runtime, auto on_value, auto on_unknown) {
    if (runtime == value_) return on_value(std::integral_constant<decltype(value_), value_> {});
    if constexpr (sizeof...(rest_) == 0) return on_unknown(runtime);
    else return dispatch_value<rest_...>(runtime, on_value, on_unknown);
}

/** Packed bits cast into no other storage: a bit file reaches the u1 shard when @p stored_dtypes_ lists one, and
 *  every other file dispatches over @p stored_dtypes_ in full. */
template <typename dataset_scalar_type_, nk_dtype_t... stored_dtypes_>
int dispatch_stored_value(nk_dtype_t dtype, auto on_dtype, auto on_unknown) {
    if constexpr (!std::is_same_v<dataset_scalar_type_, nk::u1x8_t>)
        return dispatch_value<stored_dtypes_...>(dtype, on_dtype, on_unknown);
    else if constexpr (((stored_dtypes_ == nk_u1_k) || ...))
        return dispatch_value<nk_u1_k>(dtype, on_dtype, on_unknown);
    else return on_unknown(dtype);
}

/** The stored dtypes the CPU benchmarks carry, by shrinking dynamic range. */
template <typename dataset_scalar_type_>
int dispatch_stored_dtype(nk_dtype_t dtype, auto on_dtype, auto on_unknown) {
    return dispatch_stored_value<dataset_scalar_type_, nk_f32_k, nk_bf16_k, nk_f16_k, nk_e5m2_k, nk_e4m3_k, nk_e3m2_k,
                                 nk_e2m3_k, nk_i8_k, nk_i4_k, nk_u8_k, nk_u4_k, nk_u1_k>(dtype, on_dtype, on_unknown);
}

/** The kernels one stored dtype adapts: bit-packed vectors take the bit kernels. */
template <nk_dtype_t stored_dtype_>
    requires(stored_dtype_ == nk_u1_k)
int dispatch_kernel(nk_kernel_kind_t kernel, auto on_kernel, auto on_unknown) {
    return dispatch_value<nk_kernel_dot_k, nk_kernel_hamming_k, nk_kernel_jaccard_k>(kernel, on_kernel, on_unknown);
}

/** The kernels one stored dtype adapts: every other dtype takes the metric trio. */
template <nk_dtype_t stored_dtype_>
    requires(stored_dtype_ != nk_u1_k)
int dispatch_kernel(nk_kernel_kind_t kernel, auto on_kernel, auto on_unknown) {
    return dispatch_value<nk_kernel_dot_k, nk_kernel_angular_k, nk_kernel_euclidean_k>(kernel, on_kernel, on_unknown);
}

/** The stored dtypes a flag may name, by shrinking dynamic range. */
inline constexpr nk_dtype_t stored_dtypes_k[] = {nk_f32_k,  nk_bf16_k, nk_f16_k, nk_e5m2_k, nk_e4m3_k, nk_e3m2_k,
                                                 nk_e2m3_k, nk_i8_k,   nk_i4_k,  nk_u8_k,   nk_u4_k,   nk_u1_k};

/** The distance kernels a flag may name. */
inline constexpr nk_kernel_kind_t metrics_k[] = {nk_kernel_dot_k, nk_kernel_angular_k, nk_kernel_euclidean_k,
                                                 nk_kernel_hamming_k, nk_kernel_jaccard_k};

/**
 *  @brief Every flag a benchmark binary reads, its default as the initializer, filled once in @c main.
 *
 *  @verbatim
 *  --vectors           required                Base vectors, .fbin family
 *  --queries           the first 1000 vectors  Query vectors of the same dtype
 *  --neighbors         none, recall skipped    Ground truth, .ibin
 *  --stored-dtype      the file's dtype        Stored dtype, cast on ingest
 *  --metric            angular                 Distance kernel for building and searching
 *  --connectivity      32                      Graph degree: 16, 32 or 128
 *  --expansion-add     256                     Candidate pool while inserting
 *  --expansion-search  96                      Candidate pool while searching
 *  --max-refines       0                       Upper bound on refinement passes after ingestion
 *  --expansion-refine  twice the connectivity  Candidate pool while refining
 *  --top-k             10                      Matches per query to judge
 *  --max-vectors       the whole file          Cap on ingested rows
 *  --self-recall       off                     Query every stored vector for itself
 *  --save              none                    Checkpoint name to serialize the indexes into
 *  --load              none                    Checkpoint name to restore, then insert past it
 *  --checkpoints-dir   /tmp                    Directory for checkpoint blobs
 *  --skip-vectors      0                       Rows the loaded checkpoint already covers
 *  @endverbatim
 *
 *  The CPU binaries add @ref cpu_flags and the GPU binaries their own in @c dense/vram_index_search.cuh.
 */
struct settings_t {

    /** The base vectors file, in the RetriEval `.fbin` extension family; argv-backed, so null-terminated. */
    std::string_view vectors_path;

    /** Optional query vectors of the same dtype; absent means self-queries. */
    std::string_view queries_path;

    /** Optional `.ibin` ground-truth neighbors; absent skips the quality judgment. */
    std::string_view neighbors_path;

    /** The stored dtype as named, and the representation it resolves to; unknown means "as the file". */
    std::string_view stored_dtype_name;
    nk_dtype_t stored_dtype = nk_dtype_unknown_k;

    /** The distance kernel as named, and the kernel it resolves to, driving both construction and search. */
    std::string_view metric_name;
    nk_kernel_kind_t kernel = nk_kernel_angular_k;

    /** The compute ceiling as named, and its allow-mask over NumKong kernel capabilities, intersected with the
     *  runtime detection. */
    std::string_view max_compute_name;
    nk_capability_t max_compute = nk_cap_any_k;

    /** The scheduling ceiling as named, and its allow-mask over ForkUnion scheduling capabilities, forwarded
     *  verbatim to @c fu_pool_new and mirrored by the waiter type the index instantiates. */
    std::string_view max_schedule_name;
    fu_capabilities_t max_schedule = fu_capabilities_all_k;

    /** The memory domain as named, and the engaged domains it resolves to against the machine: empty is one
     *  machine-wide heap index, a single entry pins memory and threads to it, and several build one index per
     *  listed domain. */
    std::string_view memory_domain_name;
    std::vector<fu_memory_domain_id_t> memory_domains;

    /** Worker threads; absent takes 7/8 of the available logical cores, resolved against the machine. */
    count_t threads;

    /** Staging trampoline rows per thread, so the trampoline holds this many times the thread count. */
    count_t batch_per_core {.value = 4};

    /** Neighbors-list width per node; must match a `connectivity_t` tiling. */
    count_t connectivity {.value = 32};

    /** Insertion candidate pool, calibrated on SIFT 10M against @c expansion_search for the
     *  build-once-search-forever regime. */
    count_t expansion_add {.value = 256};

    /** Search candidate pool; see @c expansion_add for the calibration. */
    count_t expansion_search {.value = 96};

    /** Matches per query to return and judge. */
    count_t top_k {.value = 10};

    /** Search every ingested vector for itself instead of the query file - the
     *  calibration-free navigability probe, with a search phase as long as the ingestion. */
    switch_t self_recall = switch_t::absent_k;

    /** Checkpoint name to serialize every domain index into after ingestion, resolved through
     *  @ref checkpoint_paths_of. */
    std::string_view save_name;

    /** Checkpoint name to restore domain indexes from before ingestion; pair it with
     *  `--skip-vectors`, so only the unindexed topping streams in. */
    std::string_view load_name;

    /** Directory the checkpoint blobs land in; point 100M-scale runs at a real disk. */
    std::string_view checkpoints_dir = "/tmp";

    /** Rows the loaded indexes already cover; ingestion resumes right after them. */
    count_t skip_vectors {.minimum = 0};

    /** Cap on ingested rows; absent ingests the whole file. */
    count_t max_vectors;

    /** First dataset row to view - slices one file across independent runs. */
    count_t first_vector {.minimum = 0};

    /** Huge pages the launcher provisioned before starting the process; zero expects none, any
     *  other value is enforced, so a base-page fallback cannot pass as a huge-page result. */
    count_t huge_pages {.minimum = 0};

    /** CUDA device group for the multi-index pipeline: one shard index per listed ordinal.
     *  Empty runs the single-device pipeline on `--device`. */
    counts_t devices {.minimum = 0};

    /** Merge the shards into one stitched graph after ingest and answer queries with a single
     *  walk; absent scatters every query to all shards and k-way merges the hits on the host. */
    switch_t merge = switch_t::absent_k;

    /** Merge candidate pool; absent inherits `--expansion-add`. */
    count_t expansion_merge;

    /** Upper bound on refinement passes after ingestion or a load; a pass that re-selects
     *  nothing ends the phase early, and zero skips it outright. */
    count_t max_refines {.minimum = 0};

    /** Refinement candidate pool; absent takes the index's default, twice the connectivity. */
    count_t expansion_refine;

    /** Rows and queries per GPU launch. */
    count_t batch {.value = 65536};

    /** Any `.fbin`-family file of every query's indexer heads, consecutive rows per query; needs `--weights`. */
    std::string_view heads_path;

    /** Any `.fbin`-family file of one row of signed head weights per query, with the model's scale folded in. */
    std::string_view weights_path;

    /** Queries the exact oracle judges: an indexer search without `--neighbors`, and attention. */
    count_t oracle_queries {.value = 16};

    /** Whether stdout is a terminal, probed once, so progress redraws one line instead of printing deciles. */
    bool terminal = false;
};

/** The flags every binary takes: the dataset, the stored dtype and metric, the graph and its refinement, the
 *  judged depth, the view, and the checkpoints. */
inline std::array<flag_t, 17> common_flags(settings_t& settings) noexcept {
    return {{
        {"--vectors", "path", "base vectors, `.fbin` family", &settings.vectors_path},
        {"--queries", "path", "query vectors, same dtype; the first 1000 vectors by default", &settings.queries_path},
        {"--neighbors", "path", "ground truth, `.ibin`", &settings.neighbors_path},
        {"--stored-dtype", "name", "stored dtype, cast on ingest; the file's dtype by default",
         &settings.stored_dtype_name},
        {"--metric", "name", "dot, angular, euclidean, hamming or jaccard; angular by default", &settings.metric_name},
        {"--connectivity", "degree", "16, 32, or 128; 32 by default", &settings.connectivity},
        {"--expansion-add", "breadth", "insertion candidate pool; 256 by default", &settings.expansion_add},
        {"--expansion-search", "breadth", "search candidate pool; 96 by default", &settings.expansion_search},
        {"--max-refines", "passes", "upper bound on whole-graph refinement passes after ingestion; 0 by default",
         &settings.max_refines},
        {"--expansion-refine", "breadth", "refinement candidate pool; twice the connectivity by default",
         &settings.expansion_refine},
        {"--top-k", "count", "matches per query to judge; 10 by default", &settings.top_k},
        {"--max-vectors", "count", "cap on ingested rows; the whole file by default", &settings.max_vectors},
        {"--self-recall", "", "query every stored vector for itself instead of the query file", &settings.self_recall},
        {"--save", "name", "checkpoint name to serialize every index into after ingestion", &settings.save_name},
        {"--load", "name", "checkpoint name to restore, then insert the rows past --skip-vectors", &settings.load_name},
        {"--checkpoints-dir", "dir", "directory for checkpoint blobs; /tmp by default", &settings.checkpoints_dir},
        {"--skip-vectors", "count", "rows the loaded checkpoint already covers; 0 by default", &settings.skip_vectors},
    }};
}

/** The flags every binary takes to rank its searches by a lightning indexer's heads instead of plain rows, and the
 *  exact oracle judging them. */
inline std::array<flag_t, 3> indexer_flags(settings_t& settings) noexcept {
    return {{
        {"--heads", "path", "any `.fbin`-family file, every query's indexer heads", &settings.heads_path},
        {"--weights", "path", "any `.fbin`-family file, one row of head weights per query", &settings.weights_path},
        {"--oracle-queries", "count", "queries the exact oracle judges without --neighbors; 16 by default",
         &settings.oracle_queries},
    }};
}

/**
 *  @brief The flags only the CPU binaries take: the ceilings, the executor, the placement, and the view.
 *
 *  @verbatim
 *  --max-compute     the newest detected       Compute ceiling by microarchitecture, or serial
 *  --max-schedule    the newest offered        Scheduling ceiling by wait instruction, or standard
 *  --threads         7/8 of the logical cores  Worker threads
 *  --memory-domain   the heap                  NUMA domain to pin to; all in dense_ram_indexes
 *  --batch-per-core  4                         Staging trampoline rows per thread
 *  --first-vector    0                         First dataset row to view
 *  --huge-pages      0                         Huge pages provisioned before launch, enforced
 *  @endverbatim
 */
inline std::array<flag_t, 7> cpu_flags(settings_t& settings) noexcept {
    return {{
        {"--max-compute", "name", "compute ceiling by microarchitecture, like `icelake`, or `serial`",
         &settings.max_compute_name},
        {"--max-schedule", "name", "scheduling ceiling by wait instruction, like `tpause`, or `standard`",
         &settings.max_schedule_name},
        {"--threads", "count", "threads; 7/8 of the logical cores by default", &settings.threads},
        {"--memory-domain", "id", "pin memory & threads to one NUMA domain; `all` in `dense_ram_indexes` only",
         &settings.memory_domain_name},
        {"--batch-per-core", "rows", "staging trampoline rows per thread; 4 by default", &settings.batch_per_core},
        {"--first-vector", "row", "first dataset row to view; 0 by default", &settings.first_vector},
        {"--huge-pages", "count", "huge pages provisioned before launch; enforced, so base-page fallbacks fail fast",
         &settings.huge_pages},
    }};
}

/** Prints `FLAG="name" does not parse, expected one of <names>` and exits with status 1. */
template <typename names_type_>
[[noreturn]] void exit_unnamed(std::string_view flag, std::string_view name, names_type_&& names) {
    fmt::println(stderr, "{}=\"{}\" does not parse, expected one of {}", flag, name, fmt::join(names, ", "));
    std::exit(1);
}

/** Resolves the names every binary shares: the stored dtype, defaulting to the file's, and the metric. Exits with
 *  status 1 on the first name that matches nothing. */
inline void resolve_common_names(settings_t& settings) {
    auto const dtype_names = stored_dtypes_k | std::views::transform(dtype_name);
    auto const metric_names = metrics_k | std::views::transform(nk_kernel_name);
    nk_dtype_t const file_dtype = utilities::dtype_of_extension(settings.vectors_path);
    if (file_dtype == nk_dtype_unknown_k || file_dtype == nk_u32_k) {
        fmt::println(
            stderr,
            "--vectors=\"{}\" does not parse, expected a .fbin, .bf16bin, .f16bin, .i8bin, .u8bin " "or .u1x8bin path",
            settings.vectors_path);
        std::exit(1);
    }
    settings.stored_dtype = file_dtype;
    if (!settings.stored_dtype_name.empty()) {
        auto const named = std::ranges::find(dtype_names, settings.stored_dtype_name);
        if (named == dtype_names.end()) exit_unnamed("--stored-dtype", settings.stored_dtype_name, dtype_names);
        settings.stored_dtype = stored_dtypes_k[named - dtype_names.begin()];
    }
    if (!settings.metric_name.empty()) {
        auto const named = std::ranges::find(metric_names, settings.metric_name);
        if (named == metric_names.end()) exit_unnamed("--metric", settings.metric_name, metric_names);
        settings.kernel = metrics_k[named - metric_names.begin()];
    }
#if USEARCH_OS_WINDOWS_
    settings.terminal = false;
#else
    settings.terminal = isatty(fileno(stdout)) != 0;
#endif
}

/** Resolves `--max-compute` into an allow-mask over NumKong kernel capabilities. Exits with status
 *  1 on a name that matches nothing. */
inline void resolve_max_compute(settings_t& settings) {
    // Ceiling names resolve into allow-masks - the native capability currency on both sides. A
    // named compute ceiling keeps its table prefix plus serial; "serial" alone keeps just the
    // fallback bit. Capping above the detection is harmless: the dispatch intersects with the
    // runtime bits anyway.
    auto compute_names = named_choices("serial", compute_choices_k,
                                       prefix_mask(compute_choices_k, std::size(compute_choices_k)));
    if (settings.max_compute_name == "serial") settings.max_compute = nk_cap_serial_k;
    else if (!settings.max_compute_name.empty()) {
        std::size_t const named = index_of_name(compute_choices_k, settings.max_compute_name);
        if (named == std::size(compute_choices_k))
            exit_unnamed("--max-compute", settings.max_compute_name, compute_names);
        settings.max_compute = nk_cap_serial_k | prefix_mask(compute_choices_k, named);
    }
}

/** Parses the CPU binaries' flags, and the ones each of @p binary_flags adds, into one @ref settings_t, resolving
 *  every name that needs no machine. On `--help` prints the usage and exits with status 0; on a bad flag prints one
 *  line and exits with status 1. */
template <typename... binary_flags_types_>
settings_t read_settings(int argc, char const* const* argv, std::string_view binary_name,
                         binary_flags_types_... binary_flags) {
    settings_t settings;
    std::vector<flag_t> flags;
    auto const append = [&](auto const& added) { flags.insert(flags.end(), added.begin(), added.end()); };
    append(common_flags(settings));
    append(cpu_flags(settings));
    append(indexer_flags(settings));
    (append(binary_flags(settings)), ...);
    parse_or_exit(argc, argv, fmt::format("{} --vectors <path> [flags]", binary_name), flags);
    resolve_common_names(settings);
    resolve_max_compute(settings);

    // A schedule ceiling caps only the waiter bits; the pool-management bits - topology, placement, huge pages -
    // stay allowed, so `fu_pool_new` keeps its full toolbox. The floor keyword is "standard", after
    // `fu::standard_yield_t` - Arm's own YIELD hint legitimately owns the "yield" row.
    auto schedule_names = named_choices("standard", schedule_choices_k,
                                        prefix_mask(schedule_choices_k, std::size(schedule_choices_k)));
    if (settings.max_schedule_name == "standard") settings.max_schedule = schedule_floor_k;
    else if (!settings.max_schedule_name.empty()) {
        std::size_t const named = index_of_name(schedule_choices_k, settings.max_schedule_name);
        if (named == std::size(schedule_choices_k))
            exit_unnamed("--max-schedule", settings.max_schedule_name, schedule_names);
        settings.max_schedule = static_cast<fu_capabilities_t>(schedule_floor_k |
                                                               prefix_mask(schedule_choices_k, named));
    }
    return settings;
}

/** Prints each setting as "- Name: value", in the grammar its flag parses from. */
inline void print(settings_t const& settings) {
    auto const or_else = [](std::string_view given, std::string_view fallback) noexcept {
        return given.empty() ? fallback : given;
    };
    auto const count_or_else = [](count_t const& count, std::string_view fallback) {
        return count.value ? fmt::format("{}", count.value) : std::string(fallback);
    };
    fmt::println("- Vectors: {}", settings.vectors_path);
    fmt::println("- Queries: {}", or_else(settings.queries_path, "the first 1000 vectors"));
    fmt::println("- Neighbors: {}", or_else(settings.neighbors_path, "none"));
    fmt::println("- Stored dtype: {}", dtype_name(settings.stored_dtype));
    fmt::println("- Metric: {}", nk_kernel_name(settings.kernel));
    fmt::println("- Max compute: {}", or_else(settings.max_compute_name, "the newest detected"));
    fmt::println("- Max schedule: {}", or_else(settings.max_schedule_name, "the newest offered"));
    fmt::println("- Memory domain: {}", or_else(settings.memory_domain_name, "the heap"));
    fmt::println("- Threads: {}", count_or_else(settings.threads, "7/8 of the logical cores"));
    fmt::println("- Batch per core: {}", settings.batch_per_core.value);
    fmt::println("- Connectivity: {}", settings.connectivity.value);
    fmt::println("- Expansion add: {}", settings.expansion_add.value);
    fmt::println("- Expansion search: {}", settings.expansion_search.value);
    fmt::println("- Max refines: {}", settings.max_refines.value);
    fmt::println("- Expansion refine: {}", count_or_else(settings.expansion_refine, "twice the connectivity"));
    fmt::println("- Top k: {}", settings.top_k.value);
    fmt::println("- Max vectors: {}", count_or_else(settings.max_vectors, "the whole file"));
    fmt::println("- First vector: {}", settings.first_vector.value);
    fmt::println("- Self recall: {}", settings.self_recall == switch_t::present_k);
    fmt::println("- Save: {}", or_else(settings.save_name, "none"));
    fmt::println("- Load: {}", or_else(settings.load_name, "none"));
    fmt::println("- Checkpoints dir: {}", settings.checkpoints_dir);
    fmt::println("- Skip vectors: {}", settings.skip_vectors.value);
    fmt::println("- Huge pages: {}", settings.huge_pages.value);
    if (!settings.heads_path.empty()) {
        fmt::println("- Heads: {}", settings.heads_path);
        fmt::println("- Weights: {}", settings.weights_path);
        fmt::println("- Oracle queries: {}", settings.oracle_queries.value);
    }
}

/** The three matrix files one checkpointed slot spans - each a RetriEval-layout dump. */
struct checkpoint_paths_t {
    std::string vectors;
    std::string neighbors;
    std::string rows;
};

/** Resolves a user-facing checkpoint name into one slot's matrix-file paths. The stamp names what
 *  the payload depends on - stored dtype, distance kernel, connectivity - so any backend
 *  instantiating the same combination computes the same names, and a mismatched one finds
 *  nothing. */
template <typename index_type_>
static checkpoint_paths_t checkpoint_paths_of(settings_t const& settings, std::string_view name, std::size_t slot,
                                              std::size_t slot_count) {
    std::string const prefix = fmt::format(
        "{}/{}-{}-{}-c{}-{}-{}", settings.checkpoints_dir, name, index_type_::scalar_t::dtype_name(),
        nk_kernel_name(index_type_::kernel_kind_k), settings.connectivity.value, slot, slot_count);
    return {fmt::format("{}.vectors.{}bin", prefix, index_type_::scalar_t::dtype_name()), //
            prefix + ".neighbors.ibin", prefix + ".rows.ibin"};
}

/** Maps the CLI neighbor-list degree onto the closed @c connectivity_t tiling set. */
inline connectivity_t connectivity_of(std::size_t degree) noexcept {
    if (degree == 16) return connectivity_16_k;
    if (degree == 32) return connectivity_32_k;
    if (degree == 128) return connectivity_128_k;
    return connectivity_unknown_k;
}

/**
 *  @brief Restores one index slot from its three RetriEval-layout matrix dumps, refilling the
 *      ID → row map the recall bookkeeping needs. Prints its own diagnostics.
 *
 *  @param[in] settings Supplies @c load_name, @c checkpoints_dir and the @c connectivity.
 *  @param[inout] index A virgin index already reserved to the run's geometry, which the mapped
 *      headers are validated against.
 *  @param[in] dimensions Row width the vectors matrix must match.
 *  @param[out] id_to_row Landing for the restored map; spans at least the checkpoint's rows.
 *  @param[in] slot This index's position among the sibling slots.
 *  @param[in] slot_count Sibling slots the save spanned; both are baked into the file names.
 *  @return True on a fully restored slot; false after printing what failed.
 */
template <typename index_type_>
static bool load_checkpoint(settings_t const& settings, index_type_& index, std::size_t dimensions,
                            std::span<u32_t> id_to_row, std::size_t slot, std::size_t slot_count) {
    using scalar_t = typename index_type_::scalar_t;
    checkpoint_paths_t const paths = checkpoint_paths_of<index_type_>(settings, settings.load_name, slot, slot_count);
    auto [vectors_mapping, vectors_map_status] = mapped_file_t::read(paths.vectors);
    auto [neighbors_mapping, neighbors_map_status] = mapped_file_t::read(paths.neighbors);
    auto [rows_mapping, rows_map_status] = mapped_file_t::read(paths.rows);
    if (failed(vectors_map_status) || failed(neighbors_map_status) || failed(rows_map_status)) {
        fmt::println("Failed to map the {} checkpoint for slot {}", settings.load_name, slot);
        return false;
    }

    auto const [vectors_extents,
                vectors_status] = utilities::parse_matrix_header<scalar_t, u32_t>(vectors_mapping.view());
    auto const [neighbors_extents,
                neighbors_status] = utilities::parse_matrix_header<u32_t, u32_t>(neighbors_mapping.view());
    auto const [map_extents, map_status] = utilities::parse_matrix_header<u32_t, u32_t>(rows_mapping.view());
    bool const parsed = succeeded(vectors_status) && succeeded(neighbors_status) && succeeded(map_status);
    std::size_t const loaded_nodes = vectors_extents.rows;
    if (!parsed || vectors_extents.columns != dimensions || neighbors_extents.columns != settings.connectivity.value ||
        map_extents.columns != 1 || neighbors_extents.rows != loaded_nodes || map_extents.rows != loaded_nodes ||
        loaded_nodes > id_to_row.size()) {
        fmt::println("Checkpoint {} disagrees with the run's geometry", settings.load_name);
        return false;
    }

    nk::matrix_view<scalar_t> const vectors_view = utilities::view_matrix_payload<scalar_t, u32_t>(
        vectors_mapping.view(), loaded_nodes, dimensions);
    nk::matrix_view<u32_t> const neighbors_view = utilities::view_matrix_payload<u32_t, u32_t>(
        neighbors_mapping.view(), loaded_nodes, settings.connectivity.value);
    // Both the chunks and the mapped dumps are dense, so a chunk is one byte move.
    std::size_t const vectors_row_bytes = static_cast<std::size_t>(vectors_view.stride_bytes(0));
    std::size_t const neighbors_row_bytes = static_cast<std::size_t>(neighbors_view.stride_bytes(0));
    bytes_view_t const vectors_source {reinterpret_cast<byte_t const*>(vectors_view.byte_data()),
                                       loaded_nodes * vectors_row_bytes};
    bytes_view_t const neighbors_source {reinterpret_cast<byte_t const*>(neighbors_view.byte_data()),
                                         loaded_nodes * neighbors_row_bytes};
    using node_id_t = typename index_type_::node_id_t;
    status_t const loaded = index.load({vectors_source, neighbors_source}, loaded_nodes,
                                       {0, static_cast<node_id_t>(loaded_nodes)});
    if (loaded != status_t::success_k) {
        fmt::println("Failed to restore {}: status {}", paths.vectors, int(loaded));
        return false;
    }
    nk::matrix_view<u32_t> const rows_view = utilities::view_matrix_payload<u32_t, u32_t>(rows_mapping.view(),
                                                                                          loaded_nodes, 1);
    std::copy_n(rows_view.data(), loaded_nodes, id_to_row.data());
    return true;
}

/**
 *  @brief Serializes one index slot as three RetriEval-layout matrices - vectors, neighbors and
 *      the ID → row map. Prints its own diagnostics.
 *
 *  @param[in] settings Supplies @c save_name, @c checkpoints_dir and the @c connectivity.
 *  @param[inout] index The built index; @c save drains it under @c immutable_k, so no insertions
 *      may be in flight.
 *  @param[in] id_to_row The map filled during ingestion; its first @c saved_shape().nodes entries
 *      ride along as the third matrix.
 *  @param[in] slot This index's position among the sibling slots.
 *  @param[in] slot_count Sibling slots the save spans; both are baked into the file names.
 *  @return The serialized byte volume, or zero on failure - a real checkpoint never is.
 */
template <typename index_type_>
static std::size_t save_checkpoint(settings_t const& settings, index_type_& index, std::span<u32_t> id_to_row,
                                   std::size_t slot, std::size_t slot_count) {
    using scalar_t = typename index_type_::scalar_t;
    using node_id_t = typename index_type_::node_id_t;
    typename index_type_::saved_shape_t const shape = index.saved_shape();
    auto vectors = nk::matrix<scalar_t>::zeros({shape.nodes, shape.dimensions}).value;
    auto neighbors = nk::matrix<node_id_t>::zeros({shape.nodes, shape.connectivity}).value;
    if (vectors.empty() || neighbors.empty()) {
        fmt::println("Failed to stage {} rows for serialization", shape.nodes);
        return 0;
    }
    // The staged matrices are dense, so a chunk is one byte move.
    std::size_t const vectors_row_bytes = static_cast<std::size_t>(vectors.stride_bytes(0));
    std::size_t const neighbors_row_bytes = static_cast<std::size_t>(neighbors.stride_bytes(0));
    bytes_span_t const vectors_target {reinterpret_cast<byte_t*>(vectors.data()), shape.nodes * vectors_row_bytes};
    bytes_span_t const neighbors_target {reinterpret_cast<byte_t*>(neighbors.data()),
                                         shape.nodes * neighbors_row_bytes};
    // One step's rows at most: a storage chunk never holds more, and the device plan ignores it.
    auto scratch =
        nk::matrix<node_id_t>::zeros({min_of(shape.nodes, power_of_two<std::size_t>(20)), shape.connectivity}).value;
    if (shape.nodes && scratch.empty()) {
        fmt::println("Failed to stage the neighbor scratch for index {}", slot);
        return 0;
    }
    status_t const saved = index.save({vectors_target, neighbors_target}, synchronization_t::immutable_k,
                                      scratch.span());
    if (saved != status_t::success_k) {
        fmt::println("Failed to serialize index {}: status {}", slot, int(saved));
        return 0;
    }

    checkpoint_paths_t const paths = checkpoint_paths_of<index_type_>(settings, settings.save_name, slot, slot_count);
    nk::matrix_view<u32_t> const rows_map_view {id_to_row.data(), {shape.nodes, 1}};
    bool const written =
        utilities::matrix_save<scalar_t>(paths.vectors.c_str(), vectors.view()) == status_t::success_k &&
        utilities::matrix_save<node_id_t>(paths.neighbors.c_str(), neighbors.view()) == status_t::success_k &&
        utilities::matrix_save<u32_t>(paths.rows.c_str(), rows_map_view) == status_t::success_k;
    if (!written) {
        fmt::println("Failed to write the {} checkpoint for slot {}", settings.save_name, slot);
        return 0;
    }
    return shape.nodes * (static_cast<std::size_t>(vectors.view().stride_bytes(0)) + //
                          shape.connectivity * sizeof(node_id_t) + sizeof(u32_t));
}

#pragma endregion

#pragma region Dataset Streaming

/** @p row widened into @p widened, through NumKong's serial casts. */
template <typename scalar_type_>
void widen_row(nk::vector_view<scalar_type_> row, std::span<f32_t> widened) noexcept {
    [[maybe_unused]] nk_status_t const cast = nk_cast_serial(row.data(), scalar_type_::dtype(), widened.data(),
                                                             nk_f32_k, widened.size(), nullptr);
}

/** @p rows in @c f32, multiplied by @p scale, landed in the stored scalar - the staging every row takes. */
template <typename scalar_type_>
std::vector<scalar_type_> stage_widened(std::vector<f32_t> rows, f32_t scale) {
    for (f32_t& value : rows) value *= scale;
    std::vector<scalar_type_> staged(rows.size());
    [[maybe_unused]] nk_status_t const cast = nk_cast_serial(rows.data(), nk_f32_k, staged.data(),
                                                             scalar_type_::dtype(), rows.size(), nullptr);
    return staged;
}

/** Every row of @p rows landed in the stored scalar under @p scale, row-major. */
template <typename scalar_type_, typename dataset_scalar_type_>
std::vector<scalar_type_> stage_rows(nk::matrix_view<dataset_scalar_type_> rows, f32_t scale) {
    std::size_t const dimensions = rows.extent(1);
    std::vector<f32_t> widened(rows.extent(0) * dimensions);
    for (std::size_t row = 0; row != rows.extent(0); ++row)
        widen_row(rows.row(row).as_vector(), std::span {widened}.subspan(row * dimensions, dimensions));
    return stage_widened<scalar_type_>(std::move(widened), scale);
}

/** NumKong's @c component_max of @p scalar_type_ - the largest magnitude one stored dimension carries - extended to
 *  the plain scalars, each its own component, whose largest is their @c finite_max. */
template <typename scalar_type_>
constexpr float component_max() noexcept {
    if constexpr (requires { scalar_type_::component_max(); }) return float(scalar_type_::component_max());
    else return float(scalar_type_::finite_max());
}

/**
 *  @brief The data-side of a run: typed views over the source rows and the bounded staging
 *      matrix the streaming @c nk::cast trampolines through, one vector at a time.
 *
 *  The staging matrix holds `--batch` rows for every thread together, never a whole-dataset
 *  copy; each thread rotates through its own contiguous area, so concurrent tasks never share a
 *  row. When the file scalar matches the stored one the staging is empty and every access is a
 *  zero-copy view into the mapping.
 */
template <typename dataset_scalar_type_, typename bench_context_type_>
struct streamed_dataset {
    using bench_t = bench_context_type_;
    using dataset_scalar_t = dataset_scalar_type_;
    using scalar_t = typename bench_context_type_::scalar_t;
    using vector_view_t = typename bench_context_type_::vector_view_t;
    static constexpr bool zero_copy_k = std::is_same_v<dataset_scalar_type_, scalar_t>;

    /** Float sources downcast into integer or byte-sized float targets only after a global
     *  symmetric rescale into the target's range; a raw cast collapses the value distribution,
     *  saturating it in integers and flushing most of an embedding to zero in fp8 and fp6. */
    static constexpr bool scaled_k = !zero_copy_k && !nk::is_integral_dtype<dataset_scalar_type_>() &&
                                     (nk::is_integral_dtype<scalar_t>() || sizeof(scalar_t) == 1);

    /** The index-side state both phases feed - punned through the pool callbacks. */
    bench_context_type_& bench;

    /** The @b [rows,dimensions] mapped source rows, in the file's own scalar. */
    nk::matrix_view<dataset_scalar_type_> vectors;

    /** The @b [queries,dimensions] mapped query rows - or a prefix of @c vectors. */
    nk::matrix_view<dataset_scalar_type_> queries;

    /** The @b [queries,depth] expected neighbors; empty when absent or when the viewed rows reshape the graph it
     *  was computed for. */
    nk::matrix_view<u32_t> ground_truth;

    /** The @b [rows] recall bookkeeping: dense node ID → global dataset row, filled while inserting. */
    std::vector<u32_t> id_to_row;

    /** The @b [threads,rows_per_thread,dimensions] staging trampoline planes. */
    nk::tensor_span<scalar_t, 3> staging;

    /** The @b [threads,rows_per_thread,dimensions] `f32` rescaling planes; only set when
     *  @c scaled_k. */
    nk::tensor_span<nk::f32_t, 3> rescaling;

    /** The global symmetric factor applied before integer downcasts. */
    float scale = 1.0f;

    /** One @c reduce_minmax pass over the source payload finds the amplitude; the same factor
     *  rescales every vector and every query, keeping their geometry consistent. A no-op unless
     *  this run downcasts a non-integral source into an integral store. */
    void establish_scale() {
        if constexpr (scaled_k) {
            // `as_vector` answers only rank-1 views, so the contiguous payload is flattened by hand.
            nk::vector_view<dataset_scalar_type_> const payload {vectors.data(), vectors.extent(0) * vectors.extent(1)};
            auto const [extremes, status] = nk::reduce_minmax<dataset_scalar_type_>(payload);
            if (nk::failed(status)) {
                fmt::println("Rescaling skipped: {}", nk::status_name(status));
                return;
            }
            float const max_amplitude = std::max(std::abs(float(extremes.min_value)),
                                                 std::abs(float(extremes.max_value)));
            // The stored dtype's largest magnitude; not a constant expression for nvcc's byte floats.
            float const target_amplitude = component_max<scalar_t>();
            if (max_amplitude > 0) scale = target_amplitude / max_amplitude;
            fmt::println("Rescaling {} → {} by ×{:.4f} before the downcast", dataset_scalar_type_::dtype_name(),
                         scalar_t::dtype_name(), scale);
        }
    }

    /** Converts one source row into the calling thread's own staging plane and views it back;
     *  @p task rotates the row within the plane, so in-flight tasks never clash. */
    vector_view_t stage(nk::vector_view<dataset_scalar_type_> source, std::size_t thread, std::size_t task) noexcept {
        std::size_t const dimensions = vectors.extent(1);
        std::size_t const rotation = task % staging.extent(1);
        auto landing = staging.row(thread).row(rotation).as_vector();
        nk::status_t status = nk::status_t::success_k;
        if constexpr (scaled_k) {
            // An f32 source feeds @c nk::scale straight from the mapping, skipping the widening cast.
            auto rescaled = rescaling.row(thread).row(rotation).as_vector();
            if constexpr (std::is_same_v<dataset_scalar_type_, nk::f32_t>)
                status = nk::scale<nk::f32_t>(source, scale, 0.0f, rescaled);
            else if (nk::succeeded(status = nk::cast<dataset_scalar_type_, nk::f32_t>(source, rescaled)))
                status = nk::scale<nk::f32_t>(rescaled, scale, 0.0f, rescaled);
            if (nk::succeeded(status)) status = nk::cast<nk::f32_t, scalar_t>(rescaled, landing);
        }
        else { status = nk::cast<dataset_scalar_type_, scalar_t>(source, landing); }
        // A failed conversion stages an empty row, which the index refuses rather than storing garbage.
        return nk::succeeded(status) ? vector_view_t {landing.data(), dimensions} : vector_view_t {};
    }
};

/** The rows a search phase answers: every ingested row under self-recall, else the query rows. */
template <typename streamed_dataset_type_>
auto searched_rows(streamed_dataset_type_ const& streamed) noexcept {
    return streamed.bench.settings.self_recall == switch_t::present_k ? streamed.vectors : streamed.queries;
}

#pragma endregion

#pragma region Process Setup

/** The machine as ForkUnion and NumKong see it: the topology and fabric every CPU binary probes, and the pool the
 *  single-index binaries spawn on it, which @ref run_cpu_bench deletes. */
struct machine_t {
    fu_topology_t topology = nullptr;
    fu_fabric_t fabric = nullptr;
    fu_pool_t pool = nullptr;
    std::size_t logical_cores = 0;
    std::size_t compute_domains = 0;
    std::size_t memory_domains = 0;

    /** Scheduling capabilities the runtime offers. */
    fu_capabilities_t schedule = fu_capabilities_unknown_k;

    /** Compute capabilities this CPU runs and this binary holds. */
    nk_capability_t compute = nk_cap_serial_k;
};

/** Probes the topology, the fabric and the capabilities; the handles stay null when ForkUnion declines. */
inline machine_t probe_machine() noexcept {
    machine_t machine;
    if (fu_topology_new(&machine.topology) != fu_success_k || fu_fabric_new(&machine.fabric) != fu_success_k)
        return machine;
    [[maybe_unused]] fu_status_t const cores_status = fu_logical_cores_count(machine.topology, &machine.logical_cores);
    [[maybe_unused]] fu_status_t const compute_status = fu_compute_domains_count(machine.topology,
                                                                                 &machine.compute_domains);
    [[maybe_unused]] fu_status_t const memory_status = fu_memory_domains_count(machine.topology,
                                                                               &machine.memory_domains);
    machine.schedule = fu_runtime_capabilities();
    if (nk_cpu_capabilities_enabled(&machine.compute) != nk_success_k) machine.compute = nk_cap_serial_k;
    return machine;
}

/** Prints the version line, then the cores, the domains and their fabric levels, and the capabilities. */
inline void print(machine_t const& machine) {
    fmt::println("USearch {}.{}.{}", USEARCH_VERSION_MAJOR, USEARCH_VERSION_MINOR, USEARCH_VERSION_PATCH);
    fmt::println("- Logical cores: {}", machine.logical_cores);
    fmt::println("- Compute domains: {}", machine.compute_domains);
    fmt::println("- Memory domains: {}", machine.memory_domains);
    for (std::size_t domain = 0; domain != machine.memory_domains; ++domain) {
        std::size_t fabric_level = 0, huge_pages_bytes = 0;
        [[maybe_unused]] fu_status_t const level_status = fu_fabric_memory_level_in(machine.fabric, domain,
                                                                                    &fabric_level);
        [[maybe_unused]] fu_status_t const pages_status = fu_volume_huge_pages_in(machine.topology, domain,
                                                                                  &huge_pages_bytes);
        fmt::println("- Memory domain {}: fabric level {}, huge pages {:.2f} GB", domain, fabric_level,
                     gigabytes(huge_pages_bytes));
    }
    char scheduling_names[256] = {0};
    std::size_t scheduling_names_length = 0;
    [[maybe_unused]] fu_status_t const named = fu_name_capabilities(machine.schedule, scheduling_names,
                                                                    sizeof(scheduling_names), &scheduling_names_length);
    fmt::println("- Scheduling: {}", scheduling_names);
    fmt::println("- Compute: {}", fmt::join(named_choices("serial", compute_choices_k, machine.compute), " "));
}

/** Bundles the launch state a CPU cascade threads through its compile-time hops: the dataset, the parsed CLI, and the
 *  machine whose pool or topology the run spawns its threads on. */
template <typename dataset_scalar_type_>
struct bench_launch {
    utilities::persisted_dataset<dataset_scalar_type_> const& dataset;
    settings_t const& settings;
    machine_t const& machine;
};

/** Whether the domains report the huge pages `--huge-pages` expects, which the launcher provisions outside the
 *  process; prints why not. */
inline bool huge_pages_provisioned(machine_t const& machine, settings_t const& settings) {
    if (!settings.huge_pages.value) return true;
    std::size_t volume = 0;
    for (std::size_t domain = 0; domain != machine.memory_domains; ++domain) {
        std::size_t domain_volume = 0;
        if (fu_volume_huge_pages_in(machine.topology, domain, &domain_volume) != fu_success_k) continue;
        volume += domain_volume;
    }
    if (!volume) fmt::println("--huge-pages expects {}, but the domains report none", settings.huge_pages.value);
    return volume != 0;
}

/** Opens the RetriEval matrix @p path in the scalar its extension names, the hop lowering the file's dtype to the C++
 *  wrapper a run is instantiated over, with the rows of @p queries_path and @p neighbors_path when named, and exits
 *  with what @p on_dataset makes of it. */
template <typename on_dataset_type_>
int open_dataset(std::string_view path, std::string_view queries_path, std::string_view neighbors_path,
                 on_dataset_type_&& on_dataset) {
    return dispatch_value<nk_f32_k, nk_bf16_k, nk_f16_k, nk_i8_k, nk_u8_k, nk_u1_k>(
        utilities::dtype_of_extension(path),
        [&](auto dtype) {
            auto [dataset, status] = utilities::persisted_dataset<typename nk::type_for<dtype.value>::type>::open(
                path, queries_path, neighbors_path);
            if (status != status_t::success_k) {
                fmt::println("Failed to open {}", path);
                return 1;
            }
            return on_dataset(dataset);
        },
        [&](auto dtype) {
            fmt::println("Unsupported dtype {} of {}", dtype_name(dtype), path);
            return 1;
        });
}

/** The process every CPU binary on one ForkUnion pool runs around its cascade: reads the flags of @p binary_name and
 *  the ones each of @p binary_flags adds, probes the machine, resolves the placement against it, spawns the pool,
 *  prints both, then opens the dataset and exits with what @p on_launch makes of its @ref bench_launch. */
template <typename on_launch_type_, typename... binary_flags_types_>
int run_cpu_bench(int argc, char const* const* argv, char const* binary_name, on_launch_type_&& on_launch,
                  binary_flags_types_... binary_flags) {
    settings_t settings = read_settings(argc, argv, binary_name, binary_flags...);
    machine_t machine = probe_machine();
    if (!machine.topology || !machine.fabric ||
        fu_pool_new(binary_name, settings.max_schedule, &machine.pool) != fu_success_k) {
        fmt::println("Failed to inspect the topology or create the pool");
        return 1;
    }

    // Resolve `--memory-domain`: empty means one machine-wide heap index, and a single ID
    // pins memory & threads to that domain. This benchmark measures exactly one index - the
    // multi-domain arrangement is the sharded `dense_ram_indexes` binaries'.
    if (!settings.memory_domain_name.empty()) {
        std::size_t domain = 0;
        char const* const end = settings.memory_domain_name.data() + settings.memory_domain_name.size();
        auto const [parsed_end, parse_status] = std::from_chars(settings.memory_domain_name.data(), end, domain);
        if (parse_status != std::errc {} || parsed_end != end || domain >= machine.memory_domains) {
            fmt::println(stderr, "--memory-domain=\"{}\" does not parse, expected a single domain below {}",
                         settings.memory_domain_name, machine.memory_domains);
            return 1;
        }
        settings.memory_domains.push_back(static_cast<fu_memory_domain_id_t>(domain));
    }

    // Default to 7/8 of the available logical cores: at full occupancy the strictly-ordered
    // publication convoy serializes on scheduler latency, and throughput collapses by an order
    // of magnitude. A single pinned domain carves the core budget from every compute domain
    // whose local memory it is - several clusters can share one NUMA node.
    bool spawned = false;
    if (settings.memory_domains.size() == 1) {
        if (!settings.threads.value) {
            std::size_t local_cores = 0;
            for (std::size_t compute_domain = 0; compute_domain != machine.compute_domains; ++compute_domain) {
                std::size_t local_memory = 0, domain_cores = 0;
                if (fu_local_memory_of(machine.topology, compute_domain, &local_memory) != fu_success_k) continue;
                if (local_memory != static_cast<std::size_t>(settings.memory_domains.front())) continue;
                if (fu_logical_cores_count_in(machine.topology, compute_domain, &domain_cores) != fu_success_k)
                    continue;
                local_cores += domain_cores;
            }
            settings.threads.value = std::max<std::size_t>(1, local_cores * 7 / 8);
        }
        spawned = fu_pool_spawn_near_memory_domain(machine.topology, machine.pool, settings.memory_domains.front(),
                                                   settings.threads.value, fu_caller_inclusive_k) == fu_success_k;
    }
    else {
        if (!settings.threads.value) settings.threads.value = std::max<std::size_t>(1, machine.logical_cores * 7 / 8);
        spawned = fu_pool_spawn(machine.topology, machine.pool, settings.threads.value, fu_caller_inclusive_k) ==
                  fu_success_k;
    }
    if (!spawned) {
        fmt::println("Failed to spawn {} threads", settings.threads.value);
        return 1;
    }
    print(machine);
    print(settings);
    if (!huge_pages_provisioned(machine, settings)) return 1;

    int const exit_code = open_dataset(
        settings.vectors_path, settings.queries_path, settings.neighbors_path,
        [&]<typename dataset_scalar_type_>(utilities::persisted_dataset<dataset_scalar_type_> const& dataset) {
            return on_launch(bench_launch<dataset_scalar_type_> {dataset, settings, machine});
        });
    fu_pool_delete(machine.pool);
    fu_fabric_delete(machine.fabric);
    fu_topology_delete(machine.topology);
    return exit_code;
}

#pragma endregion

} // namespace unum::usearch::bench
