/**
 *  @file bench/dense.cuh
 *  @author Ash Vardanian
 *  @date July 22, 2026
 *  @brief The backend-neutral benchmarking surface shared by every dense-index residency - volatile
 *      in-RAM, GPU-resident, and persisted alike.
 *
 *  Only what every binary shares lives here: the CLI option set, the SafeGraphs checkpoint naming
 *  and the single-index monolith, the dataset streaming trampoline, and the process setup of the
 *  CPU binaries. The search telemetry and judgment live in @c dense_search.cuh, the algebra surface
 *  in @c dense_algebra.cuh. The run loops are backend property, so this header names no index and
 *  reads no index-specific members - the same surface compiles under host compilers and nvcc.
 */
#pragma once

#include <cmath>   // `std::abs` for the rescale amplitude
#include <cstdio>  // `stdout` for the TTY probe
#include <cstdlib> // `std::exit`
#include <cstring> // `std::memcpy` - the checkpoint runs

#include <algorithm>    // `std::ranges::find`
#include <charconv>     // `std::from_chars` - the `--memory-domain` ordinal
#include <filesystem>   // `std::filesystem::temp_directory_path` - the default checkpoints dir
#include <ranges>       // `std::views::iota`, `std::views::filter`, `std::views::transform`
#include <span>         // `std::span`
#include <string>       // `std::string`
#include <string_view>  // `std::string_view`
#include <system_error> // `std::error_code`
#include <type_traits>  // `std::integral_constant` for the dispatch chains
#include <vector>       // `std::vector`

#include <fmt/format.h>
#include <fmt/ranges.h> // `fmt::join`

#include <forkunion/forkunion.h>
#include <forkunion/capabilities.hpp> // `fu::preferred_cache_hints_t` - the engines' line hints
#include <forkunion/topology.hpp>     // `fu::cache_line_bytes` - the line they step by
#include <numkong/numkong.h>
#include <numkong/numkong.hpp>
#include <stringzilla/stringzilla.hpp>

#include "usearch/utilities/safegraphs_format.hpp"
#include "usearch/dense/types.hpp"             // `dense::dtype_of` - the checkpoint columns
#include "usearch/dense/shard_checkpoints.hpp" // `dense::norms_column_of`, `dense::monolith_header` - the norms column

#if !USEARCH_OS_WINDOWS_
#include <unistd.h> // `isatty`
#endif

#include "abi.hpp" // `flag_t`, `count_t`, `counts_t`, `switch_t`, `parse_or_exit`, `mapped_rows_t`
#include "bench.hpp"

namespace unum::usearch::bench {
using tools::count_t;
using tools::counts_t;
using tools::flag_t;
using tools::parse_or_exit;
using tools::switch_t;

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
#if FORKUNION_ARCH_X8664_
    {"pause", fu_capability_x86_pause_k},
    {"tpause", fu_capability_x86_tpause_k},
#elif FORKUNION_ARCH_ARM64_
    {"yield", fu_capability_arm64_yield_k},
    {"wfet", fu_capability_arm64_wfet_k},
#elif FORKUNION_ARCH_RISCV64_
    {"pause", fu_capability_riscv64_pause_k},
    {"wrs", fu_capability_riscv64_wrs_k},
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

/** The line the benchmarked engines prefetch by, as `src/` hands it them: this machine's cache
 *  line, or the default alignment where no platform source names one. */
inline std::size_t machine_cache_line_bytes() noexcept {
    std::size_t const machine_line_bytes = fu::cache_line_bytes();
    return machine_line_bytes ? machine_line_bytes : default_alignment_k;
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

/** Whether the build compiles @p dtype as storage: the first `USEARCH_DENSE_DTYPES` rungs of the
 *  library's ladder, and the f32 reference only the benchmarks store. */
constexpr bool stored_dtype_compiled(nk_dtype_t dtype) noexcept {
    switch (dtype) {
    case nk_f32_k:
    case nk_bf16_k: return true;
    case nk_u1_k: return USEARCH_DENSE_DTYPES >= 2;
    case nk_i8_k: return USEARCH_DENSE_DTYPES >= 3;
    case nk_e4m3_k: return USEARCH_DENSE_DTYPES >= 4;
    case nk_f16_k: return USEARCH_DENSE_DTYPES >= 5;
    case nk_u8_k: return USEARCH_DENSE_DTYPES >= 6;
    case nk_e5m2_k: return USEARCH_DENSE_DTYPES >= 7;
    case nk_i4_k: return USEARCH_DENSE_DTYPES >= 8;
    case nk_u4_k: return USEARCH_DENSE_DTYPES >= 9;
    case nk_e3m2_k: return USEARCH_DENSE_DTYPES >= 10;
    case nk_e2m3_k: return USEARCH_DENSE_DTYPES >= 11;
    default: return false;
    }
}

/** Packed bits cast into no other storage: a bit file reaches the u1 shard when @p stored_dtypes_
 *  lists one, and every other file dispatches over @p stored_dtypes_ in full; a dtype the build
 *  leaves out takes @p on_unknown. */
template <typename dataset_scalar_type_, nk_dtype_t... stored_dtypes_>
int dispatch_stored_value(nk_dtype_t dtype, auto on_dtype, auto on_unknown) {
    auto on_compiled = [&](auto stored) {
        if constexpr (stored_dtype_compiled(decltype(stored)::value)) return on_dtype(stored);
        else return on_unknown(decltype(stored)::value);
    };
    if constexpr (!std::is_same_v<dataset_scalar_type_, nk::u1x8_t>)
        return dispatch_value<stored_dtypes_...>(dtype, on_compiled, on_unknown);
    else if constexpr (((stored_dtypes_ == nk_u1_k) || ...))
        return dispatch_value<nk_u1_k>(dtype, on_compiled, on_unknown);
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
 *  Each flag also reads its default from its @c RETRIEVAL_ variable, like @c RETRIEVAL_TOP_K for
 *  @c --top-k.
 *
 *  @verbatim
 *  --base-vectors      required                     Base vectors, .fbin family
 *  --query-vectors     the first 1000 base vectors  Query vectors of the same dtype
 *  --query-neighbors   none, recall skipped         Ground truth, .ibin
 *  --stored-dtype      the file's dtype             Stored dtype, cast on ingest
 *  --metric            angular                      Distance kernel for building and searching
 *  --connectivity      32                           Graph degree: 16, 32 or 128
 *  --expansion-add     256                          Candidates while inserting
 *  --expansion-search  96                           Candidates while searching
 *  --max-refines       0                            Upper bound on refinement passes after ingestion
 *  --expansion-refine  twice the connectivity       Candidates while refining
 *  --top-k             10                           Matches per query to judge
 *  --first-vector      0                            First dataset row viewed, keyed by its row
 *  --max-base-vectors  the whole file               Cap on ingested rows
 *  --self-search       off                          Query every stored vector for itself
 *  --save              none                         Checkpoint name to serialize the indexes into
 *  --load              none                         Checkpoint name to restore, then insert past it
 *  --checkpoints-dir   the temporary directory      Directory for checkpoint blobs
 *  --skip-vectors      0                            Rows the loaded checkpoint already covers
 *  --inserts-per-core  2048                         Rows per insert call for each thread or multiprocessor
 *  --queries-per-core  2048                         Queries per search call for each thread or multiprocessor
 *  @endverbatim
 *
 *  The CPU binaries add @ref cpu_flags and the GPU binaries their own in @c dense/vram_index_search.cuh.
 */
struct settings_t {

    /** The base vectors file, in the RetriEval `.fbin` extension family; argv-backed, so null-terminated. */
    std::string_view base_vectors_path;

    /** Optional query vectors of the same dtype; absent means self-queries. */
    std::string_view query_vectors_path;

    /** Optional `.ibin` ground-truth neighbors; absent skips the quality judgment. */
    std::string_view query_neighbors_path;

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

    /** Rows each insert call takes per core the run drives, after StringWars' per-core batches. */
    count_t inserts_per_core {.value = 2048};

    /** Queries each search call takes per core the run drives. */
    count_t queries_per_core {.value = 2048};

    /** Cores the run drives, resolved at launch: the threads on CPU, the engaged devices'
     *  multiprocessors on GPU. */
    std::size_t cores = 0;

    /** Neighbors-list width per node; must match a `connectivity_t` tiling. */
    count_t connectivity {.value = 32};

    /** How many candidates an insertion keeps, calibrated on SIFT 10M against @c expansion_search
     *  for the build-once-search-forever regime. */
    count_t expansion_add {.value = 256};

    /** How many candidates a search keeps; see @c expansion_add for the calibration. */
    count_t expansion_search {.value = 96};

    /** Matches per query to return and judge. */
    count_t top_k {.value = 10};

    /** Search every ingested vector for itself instead of the query file - the
     *  calibration-free navigability probe, with a search phase as long as the ingestion. */
    switch_t self_search = switch_t::absent_k;

    /** Answer every query by an exact scan of every stored vector instead of a walk. */
    switch_t scan = switch_t::absent_k;

    /** Checkpoint name to serialize every domain index into after ingestion, resolved through
     *  @ref checkpoint_path_of. */
    std::string_view save_name;

    /** Checkpoint name to restore domain indexes from before ingestion; pair it with
     *  `--skip-vectors`, so only the unindexed topping streams in. */
    std::string_view load_name;

    /** Directory the checkpoint blobs land in; point 100M-scale runs at a real disk. */
    std::string_view checkpoints_dir;

    /** Rows the loaded indexes already cover; ingestion resumes right after them. */
    count_t skip_vectors {.minimum = 0};

    /** Cap on ingested rows; absent ingests the whole file. */
    count_t max_base_vectors;

    /** First dataset row to view, keyed by its row - slices one file across independent runs. */
    count_t first_vector {.minimum = 0};

    /** Free 1 GiB huge pages the host must hold at launch; zero expects none, any other value is
     *  enforced, so a base-page fallback cannot pass as a huge-page result. */
    count_t huge_pages {.minimum = 0};

    /** Ingest the second half of the rows into a donor index and absorb it into the first, so the
     *  absorbing refine is measured on its own; absent ingests every row into one index. */
    switch_t merge = switch_t::absent_k;

    /** How many candidates a merge keeps; absent inherits `--expansion-add`. */
    count_t expansion_merge;

    /** Indexes the inserted rows build as, over equal contiguous slices, the later ones absorbed
     *  into the first; absent builds one. */
    count_t donors;

    /** Entries one shard may hold, past which an insertion opens another shard generation; absent
     *  bounds shards by the ID space alone. */
    count_t shard_capacity;

    /** Entries one shard may hold after the built index compacts in place; absent skips the
     *  compaction. */
    count_t compact_capacity;

    /** Upper bound on refinement passes after ingestion or a load; a pass that re-selects
     *  nothing ends the phase early, and zero skips it outright. */
    count_t max_refines {.minimum = 0};

    /** How many candidates a refinement keeps; absent takes the insertion's, since a pass repeats
     *  insertion's selection with the whole graph visible. */
    count_t expansion_refine;

    /** Any `.fbin`-family file of every query's indexer heads, consecutive rows per query; needs `--weights`. */
    std::string_view heads_path;

    /** Any `.fbin`-family file of one row of signed head weights per query, with the model's scale folded in. */
    std::string_view weights_path;

    /** Queries the exact oracle judges: an indexer search without `--query-neighbors`, and
     *  attention. */
    count_t oracle_queries {.value = 16};

    /** Whether stdout is a terminal, probed once, so progress redraws one line instead of printing deciles. */
    bool terminal = false;
};

/** The flags every binary takes: the dataset, the stored dtype and metric, the graph and its refinement, the
 *  judged depth, the view, and the checkpoints. */
inline std::array<flag_t, 21> common_flags(settings_t& settings) noexcept {
    return {{
        {"--base-vectors", "path", "base vectors, `.fbin` family", &settings.base_vectors_path},
        {"--query-vectors", "path", "query vectors, same dtype; the first 1000 base vectors by default",
         &settings.query_vectors_path},
        {"--query-neighbors", "path", "ground truth, `.ibin`", &settings.query_neighbors_path},
        {"--stored-dtype", "name", "stored dtype, cast on ingest; the file's dtype by default",
         &settings.stored_dtype_name},
        {"--metric", "name", "dot, angular, euclidean, hamming or jaccard; angular by default", &settings.metric_name},
        {"--connectivity", "degree", "16, 32, or 128; 32 by default", &settings.connectivity},
        {"--expansion-add", "breadth", "insertion candidates; 256 by default", &settings.expansion_add},
        {"--expansion-search", "breadth", "search candidates; 96 by default", &settings.expansion_search},
        {"--max-refines", "passes", "upper bound on whole-graph refinement passes after ingestion; 0 by default",
         &settings.max_refines},
        {"--expansion-refine", "breadth", "refinement candidates; --expansion-add by default",
         &settings.expansion_refine},
        {"--top-k", "count", "matches per query to judge; 10 by default", &settings.top_k},
        {"--first-vector", "row", "first dataset row viewed, keyed by its row; 0 by default", &settings.first_vector},
        {"--max-base-vectors", "count", "cap on ingested rows; the whole file by default", &settings.max_base_vectors},
        {"--self-search", "", "query every stored vector for itself instead of the query file", &settings.self_search},
        {"--scan", "", "answer every query by an exact scan of every stored row instead of a walk", &settings.scan},
        {"--save", "name", "checkpoint name to serialize every index into after ingestion", &settings.save_name},
        {"--load", "name", "checkpoint name to restore, then insert the rows past --skip-vectors", &settings.load_name},
        {"--checkpoints-dir", "dir", "directory for checkpoint blobs; the temporary directory by default",
         &settings.checkpoints_dir},
        {"--skip-vectors", "count", "rows the loaded checkpoint already covers; 0 by default", &settings.skip_vectors},
        {"--inserts-per-core", "rows", "rows per insert call, per CPU thread or GPU multiprocessor; 2048 by default",
         &settings.inserts_per_core},
        {"--queries-per-core", "rows", "queries per search call, per CPU thread or GPU multiprocessor; 2048 by default",
         &settings.queries_per_core},
    }};
}

/** Rows each insert call takes: `--inserts-per-core` for every core the run drives, capped by
 *  @p total_rows. */
inline std::size_t insert_batch_rows(settings_t const& settings, std::size_t total_rows) noexcept {
    return min_of(settings.inserts_per_core.value * settings.cores, total_rows);
}

/** Queries each search call takes: `--queries-per-core` for every core the run drives, capped by
 *  @p total_rows. */
inline std::size_t query_batch_rows(settings_t const& settings, std::size_t total_rows) noexcept {
    return min_of(settings.queries_per_core.value * settings.cores, total_rows);
}

/** The flags every binary takes to rank its searches by a lightning indexer's heads instead of plain rows, and the
 *  exact oracle judging them. */
inline std::array<flag_t, 3> indexer_flags(settings_t& settings) noexcept {
    return {{
        {"--heads", "path", "any `.fbin`-family file, every query's indexer heads", &settings.heads_path},
        {"--weights", "path", "any `.fbin`-family file, one row of head weights per query", &settings.weights_path},
        {"--oracle-queries", "count", "queries the exact oracle judges without --query-neighbors; 16 by default",
         &settings.oracle_queries},
    }};
}

/**
 *  @brief The flags only the CPU binaries take: the ceilings, the executor, and the placement.
 *
 *  @verbatim
 *  --max-compute     the newest detected       Compute ceiling by microarchitecture, or serial
 *  --max-schedule    the newest offered        Scheduling ceiling by wait instruction, or standard
 *  --threads         7/8 of the logical cores  Worker threads
 *  --memory-domain   the heap                  NUMA domain to pin to; all in dense_ram_indexes
 *  --huge-pages      0                         Free 1 GiB huge pages required at launch, enforced
 *  @endverbatim
 */
inline std::array<flag_t, 5> cpu_flags(settings_t& settings) noexcept {
    return {{
        {"--max-compute", "name", "compute ceiling by microarchitecture, like `icelake`, or `serial`",
         &settings.max_compute_name},
        {"--max-schedule", "name", "scheduling ceiling by wait instruction, like `tpause`, or `standard`",
         &settings.max_schedule_name},
        {"--threads", "count", "threads; 7/8 of the logical cores by default", &settings.threads},
        {"--memory-domain", "id", "pin memory & threads to one NUMA domain; `all` in `dense_ram_indexes` only",
         &settings.memory_domain_name},
        tools::huge_pages_flag(settings.huge_pages),
    }};
}

/** Prints `FLAG="name" does not parse, expected one of <names>` and exits with status 1. */
template <typename names_type_>
[[noreturn]] void exit_unnamed(std::string_view flag, std::string_view name, names_type_&& names) {
    fmt::println(stderr, "{}=\"{}\" does not parse, expected one of {}", flag, name, fmt::join(names, ", "));
    std::exit(1);
}

/** The NumKong dtype of the vectors a RetriEval @p path holds, by its extension, or unknown. */
inline nk_dtype_t dtype_of_extension(std::string_view path) noexcept {
    switch (tools::scalar_of_extension(path)) {
    case usearch_scalar_f32_k: return nk_f32_k;
    case usearch_scalar_bf16_k: return nk_bf16_k;
    case usearch_scalar_f16_k: return nk_f16_k;
    case usearch_scalar_i8_k: return nk_i8_k;
    case usearch_scalar_u8_k: return nk_u8_k;
    case usearch_scalar_u1x8_k: return nk_u1_k;
    default: return nk_dtype_unknown_k;
    }
}

/** Resolves the names every binary shares: the stored dtype, defaulting to the file's, and the metric. Exits with
 *  status 1 on the first name that matches nothing. */
inline void resolve_common_names(settings_t& settings) {
    if (!settings.expansion_refine.value) settings.expansion_refine.value = settings.expansion_add.value;
    auto const dtype_names = stored_dtypes_k | std::views::transform(dtype_name);
    auto const metric_names = metrics_k | std::views::transform(nk_kernel_name);
    nk_dtype_t const file_dtype = dtype_of_extension(settings.base_vectors_path);
    if (file_dtype == nk_dtype_unknown_k) {
        fmt::println(stderr, "--base-vectors=\"{}\" does not parse, expected a {} path", settings.base_vectors_path,
                     ".fbin, .bf16bin, .f16bin, .i8bin, .u8bin, .u1x8bin or .b1bin");
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
    parse_or_exit(argc, argv, fmt::format("{} --base-vectors <path> [flags]", binary_name), flags);
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
    fmt::println("- Base vectors: {}", settings.base_vectors_path);
    fmt::println("- Query vectors: {}", or_else(settings.query_vectors_path, "the first 1000 base vectors"));
    fmt::println("- Query neighbors: {}", or_else(settings.query_neighbors_path, "none"));
    fmt::println("- Stored dtype: {}", dtype_name(settings.stored_dtype));
    fmt::println("- Metric: {}", nk_kernel_name(settings.kernel));
    fmt::println("- Max compute: {}", or_else(settings.max_compute_name, "the newest detected"));
    fmt::println("- Max schedule: {}", or_else(settings.max_schedule_name, "the newest offered"));
    fmt::println("- Memory domain: {}", or_else(settings.memory_domain_name, "the heap"));
    fmt::println("- Threads: {}", count_or_else(settings.threads, "7/8 of the logical cores"));
    fmt::println("- Inserts per core: {}", settings.inserts_per_core.value);
    fmt::println("- Queries per core: {}", settings.queries_per_core.value);
    fmt::println("- Connectivity: {}", settings.connectivity.value);
    fmt::println("- Expansion add: {}", settings.expansion_add.value);
    fmt::println("- Expansion search: {}", settings.expansion_search.value);
    fmt::println("- Max refines: {}", settings.max_refines.value);
    fmt::println("- Expansion refine: {}", settings.expansion_refine.value);
    fmt::println("- Top k: {}", settings.top_k.value);
    fmt::println("- Max base vectors: {}", count_or_else(settings.max_base_vectors, "the whole file"));
    fmt::println("- First vector: {}", settings.first_vector.value);
    fmt::println("- Self search: {}", settings.self_search == switch_t::present_k);
    fmt::println("- Scan: {}", settings.scan == switch_t::present_k);
    fmt::println("- Save: {}", or_else(settings.save_name, "none"));
    fmt::println("- Load: {}", or_else(settings.load_name, "none"));
    fmt::println("- Checkpoints dir: {}", or_else(settings.checkpoints_dir, "the temporary directory"));
    fmt::println("- Skip vectors: {}", settings.skip_vectors.value);
    fmt::println("- Huge pages: {}", settings.huge_pages.value);
    if (!settings.heads_path.empty()) {
        fmt::println("- Heads: {}", settings.heads_path);
        fmt::println("- Weights: {}", settings.weights_path);
        fmt::println("- Oracle queries: {}", settings.oracle_queries.value);
    }
}

/** The directory checkpoints land in: `--checkpoints-dir`, else the temporary one, which follows
 *  `TMPDIR`. */
inline std::filesystem::path checkpoints_folder(settings_t const& settings) {
    std::error_code error;
    if (settings.checkpoints_dir.empty()) return std::filesystem::temp_directory_path(error);
    return std::filesystem::path(settings.checkpoints_dir);
}

/** Resolves a user-facing checkpoint name into its SafeGraphs path: one monolith for a single index
 *  or a lone shard, the manifest beside one monolith per shard otherwise. The stamp names what the
 *  payload depends on - stored dtype, distance kernel, connectivity - so any backend instantiating
 *  the same combination computes the same name, and a mismatched one finds nothing. */
template <typename index_type_>
std::string checkpoint_path_of(settings_t const& settings, index_type_ const& index, std::string_view name) {
    std::string const file = fmt::format( //
        "{}-{}-{}-c{}.safegraphs", name, index_type_::scalar_t::dtype_name(), nk_kernel_name(index.kernel()),
        settings.connectivity.value);
    return (checkpoints_folder(settings) / file).string();
}

/** The norms column dtype a single index's checkpoint carries: the dot product's own token where
 *  the index keeps norms, @c unknown_k where it keeps none. */
template <typename index_type_>
dtype_t checkpoint_norms_dtype(index_type_ const& index) noexcept {
    using dot_result_t = typename index_type_::scalar_t::dot_result_t;
    return index.stores_norms() ? usearch::dense::dtype_of(dot_result_t::dtype()) : dtype_t::unknown_k;
}

/** Maps the CLI neighbor-list degree onto the closed @c connectivity_t tiling set. */
inline connectivity_t connectivity_of(std::size_t degree) noexcept {
    if (degree == 16) return connectivity_16_k;
    if (degree == 32) return connectivity_32_k;
    if (degree == 128) return connectivity_128_k;
    return connectivity_unknown_k;
}

/**
 *  @brief Restores a single index from a SafeGraphs checkpoint - its monolith, or else the manifest
 *      a sharded save committed, whose parts append in order - and refills the ID → row map from
 *      the keys. Prints its own diagnostics.
 *
 *  @param[in] settings Supplies @c load_name and @c checkpoints_dir.
 *  @param[in,out] index A virgin index already reserved to the run's geometry, which every part's
 *      header is vetted against.
 *  @param[out] id_to_row Output for the restored map; spans at least the checkpoint's rows.
 *  @param[in] stream The executor's stream a device index restores on; a RAM index takes none.
 *  @return True on a fully restored index; false after printing what failed.
 */
template <typename index_type_, typename... stream_types_>
static bool load_checkpoint(settings_t const& settings, index_type_& index, std::span<u64_t> id_to_row,
                            stream_types_ const&... stream) {
    using scalar_t = typename index_type_::scalar_t;
    using node_id_t = typename index_type_::node_id_t;
    typename index_type_::saved_shape_t const shape = index.saved_shape();
    std::string const monolith_path = checkpoint_path_of(settings, index, settings.load_name);
    std::vector<std::string> part_paths;
    auto per_part = [&](safegraphs_part_t part) {
        part_paths.push_back((checkpoints_folder(settings) / part.path).string());
    };
    if (failed(safegraphs_read_manifest(monolith_path, std::allocator<byte_t> {}, per_part).status))
        part_paths = {monolith_path};

    std::size_t loaded = 0;
    for (std::string const& path : part_paths) {
        auto [mapping, map_status] = mapped_file_t::read(path);
        if (failed(map_status)) {
            fmt::println("Failed to map {}: {}", path, status_to_string(map_status));
            return false;
        }
        bytes_view_t const monolith = mapping.view();
        growing_buffer<safegraphs_named_column_t, std::allocator<byte_t>> extra_columns {std::allocator<byte_t> {}};
        auto const [header, header_status] = safegraphs_read_header(monolith, std::allocator<byte_t> {},
                                                                    dummy_callback_t {}, &extra_columns);
        std::size_t const nodes = header.nodes();
        if (failed(header_status) || header.key.offsets || header.value.offsets || header.neighbors.offsets ||
            !safegraphs_key_widens(header.key.dtype, dtype_t::u64_k) ||
            header.value.dtype != usearch::dense::dtype_of(scalar_t::dtype()) ||
            header.neighbors.dtype != dtype_t::u32_k || regular_width(header.value.shape) != shape.dimensions ||
            regular_width(header.neighbors.shape) != shape.connectivity || header.monolith_bytes() > monolith.size() ||
            nodes > id_to_row.size() - loaded) {
            fmt::println("Checkpoint {} disagrees with the run's geometry", path);
            return false;
        }
        // A `u32` key column zero-extends into the `u64` rows, which the index files as its keys.
        byte_t const* const key_column = header.data_of(monolith, header.key).data();
        std::size_t const key_bytes = dtype_bytes(header.key.dtype);
        std::vector<typename index_type_::key_t> keys(nodes);
        for (std::size_t node = 0; node != nodes; ++node) {
            id_to_row[loaded + node] = safegraphs_widened_key<u64_t>(key_column + node * key_bytes, header.key.dtype);
            keys[node] = uuid_of(id_to_row[loaded + node]);
        }
        // Each part's IDs are its own, so it appends as one source past every earlier part.
        status_t const restored = index.load(
            {header.data_of(monolith, header.value), header.data_of(monolith, header.neighbors),
             usearch::dense::norms_column_of(header, monolith, checkpoint_norms_dtype(index))},
            {keys.data(), nodes}, nodes, {0, static_cast<node_id_t>(nodes)}, stream...);
        if (failed(restored)) {
            fmt::println("Failed to restore {}: {}", path, status_to_string(restored));
            return false;
        }
        loaded += nodes;
    }
    return true;
}

/**
 *  @brief Serializes a single index as one SafeGraphs monolith whose keys are the ID → row map, so
 *      every residency, sharded or not, and the NVMe benches read it back. Prints its own
 *      diagnostics.
 *
 *  @param[in] settings Supplies @c save_name, @c checkpoints_dir and the @c connectivity.
 *  @param[in,out] index The built index; @c save drains it under @c immutable_k, so no insertions
 *      may be in flight.
 *  @param[in] id_to_row The map filled during ingestion; its first @c saved_shape().nodes entries
 *      become the key column.
 *  @param[in] scale The factor the rows were rescaled by, which a view stages its queries under.
 *  @param[in] stream The executor's stream a device index serializes on; a RAM index takes none.
 *  @return The monolith's bytes, or zero on failure - a real checkpoint never is.
 */
template <typename index_type_, typename... stream_types_>
static std::size_t save_checkpoint(settings_t const& settings, index_type_& index, std::span<u64_t const> id_to_row,
                                   float scale, stream_types_ const&... stream) {
    using scalar_t = typename index_type_::scalar_t;
    typename index_type_::saved_shape_t const shape = index.saved_shape();
    std::string const path = checkpoint_path_of(settings, index, settings.save_name);
    std::string const scale_text = fmt::format("{}", scale);
    safegraphs_custom_field_t const fields[] = {{"metric", nk_kernel_name(index.kernel())}, {"scale", scale_text}};
    safegraphs_named_column_t norms;
    auto const [header, header_status] = usearch::dense::monolith_header(
        {shape.nodes, dtype_t::u64_k, usearch::dense::dtype_of(scalar_t::dtype()), shape.dimensions, dtype_t::u32_k,
         shape.connectivity, checkpoint_norms_dtype(index)},
        safegraphs_alignment_default_k, fields, norms);
    if (failed(header_status) || id_to_row.size() < shape.nodes) {
        fmt::println("Failed to frame {} rows for {}", shape.nodes, path);
        return 0;
    }
    auto [file, map_status] = mapped_file_t::write(path, header.monolith_bytes());
    if (failed(map_status)) {
        fmt::println("Failed to map {}", path);
        return 0;
    }
    bytes_span_t const monolith = file.span();
    status_t const saved = index.save({header.rows_of(monolith, header.value, 0, shape.nodes),
                                       header.rows_of(monolith, header.neighbors, 0, shape.nodes),
                                       header.rows_of(monolith, norms.column, 0, shape.nodes)},
                                      synchronization_t::immutable_k, stream...);
    if (failed(saved)) {
        fmt::println("Failed to serialize the index: {}", status_to_string(saved));
        return 0;
    }
    std::memcpy(header.rows_of(monolith, header.key, 0, shape.nodes).data(), id_to_row.data(),
                shape.nodes * sizeof(u64_t));
    if (failed(safegraphs_write_header(monolith, header, fields)) || failed(file.sync())) {
        fmt::println("Failed to commit {}", path);
        return 0;
    }
    return header.monolith_bytes();
}

#pragma endregion

#pragma region Dataset Streaming

/** @p row widened into @p widened, through NumKong's serial casts. */
template <typename scalar_type_>
void widen_row(nk::vector_view<scalar_type_> row, std::span<f32_t> widened) noexcept {
    [[maybe_unused]] nk_status_t const cast = nk_cast_serial(row.data(), scalar_type_::dtype(), widened.data(),
                                                             nk_f32_k, widened.size(), nullptr);
}

/** @p rows in @c f32, multiplied by @p scale and cast to the stored scalar, which is the staging
 *  every row goes through. */
template <typename scalar_type_>
std::vector<scalar_type_> stage_widened(std::vector<f32_t> rows, f32_t scale) {
    for (f32_t& value : rows) value *= scale;
    std::vector<scalar_type_> staged(rows.size());
    [[maybe_unused]] nk_status_t const cast = nk_cast_serial(rows.data(), nk_f32_k, staged.data(),
                                                             scalar_type_::dtype(), rows.size(), nullptr);
    return staged;
}

/** Every row of @p rows cast to the stored scalar under @p scale, row-major. */
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
 *  The staging matrix holds one batched call's rows, or a row per thread for one-row calls, never a
 *  whole-dataset copy; each thread rotates through its own contiguous area, so concurrent tasks
 *  never share a row. When the file scalar matches the stored one the staging is empty and every
 *  access is a zero-copy view into the mapping.
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
    std::vector<u64_t> id_to_row;

    /** The @b [threads,rows_per_thread,dimensions] staging trampoline rows. */
    nk::tensor_span<scalar_t, 3> staging;

    /** The @b [threads,rows_per_thread,dimensions] `f32` rescaling rows; only set when
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
            float const max_amplitude = max_of(std::abs(float(extremes.min_value)),
                                               std::abs(float(extremes.max_value)));
            // The stored dtype's largest magnitude; not a constant expression for nvcc's byte floats.
            float const target_amplitude = component_max<scalar_t>();
            if (max_amplitude > 0) scale = target_amplitude / max_amplitude;
            fmt::println("Rescaling {} → {} by ×{:.4f} before the downcast", dataset_scalar_type_::dtype_name(),
                         scalar_t::dtype_name(), scale);
        }
    }

    /** Converts one source row into the calling thread's own staging rows and views it back;
     *  @p task rotates the row among them, so in-flight tasks never clash. */
    vector_view_t stage(nk::vector_view<dataset_scalar_type_> source, std::size_t thread, std::size_t task) noexcept {
        std::size_t const dimensions = vectors.extent(1);
        std::size_t const rotation = task % staging.extent(1);
        auto staged = staging.row(thread).row(rotation).as_vector();
        nk::status_t status = nk::status_t::success_k;
        if constexpr (scaled_k) {
            // An f32 source feeds @c nk::scale straight from the mapping, skipping the widening cast.
            auto rescaled = rescaling.row(thread).row(rotation).as_vector();
            if constexpr (std::is_same_v<dataset_scalar_type_, nk::f32_t>)
                status = nk::scale<nk::f32_t>(source, scale, 0.0f, rescaled);
            else if (nk::succeeded(status = nk::cast<dataset_scalar_type_, nk::f32_t>(source, rescaled)))
                status = nk::scale<nk::f32_t>(rescaled, scale, 0.0f, rescaled);
            if (nk::succeeded(status)) status = nk::cast<nk::f32_t, scalar_t>(rescaled, staged);
        }
        else { status = nk::cast<dataset_scalar_type_, scalar_t>(source, staged); }
        // A failed conversion stages an empty row, which the index refuses rather than storing garbage.
        return nk::succeeded(status) ? vector_view_t {staged.data(), dimensions} : vector_view_t {};
    }
};

/** The rows a search phase answers: every ingested row under self-search, else the query rows. */
template <typename streamed_dataset_type_>
auto searched_rows(streamed_dataset_type_ const& streamed) noexcept {
    return streamed.bench.settings.self_search == switch_t::present_k ? streamed.vectors : streamed.queries;
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

/**
 *  @brief A RetriEval triple viewed as @p scalar_type_ rows in place: the base vectors, a glob of
 *      parts numbering their rows on across each other, the queries, and the ground truth's @c u32
 *      rows, widened only where the judgment compares them.
 *
 *  Queries and ground truth are optional: without queries a run searches its own rows, and without
 *  ground truth it skips the judgment against it.
 */
template <typename scalar_type_>
struct mapped_dataset {
    tools::mapped_rows_t base, query_rows, neighbor_rows;

    /** Opens the triple, failing on a dimensionality mismatch between the base and the queries, or
     *  a row-count mismatch between the queries and the ground truth. */
    static expected<mapped_dataset> open(std::string_view vectors_path, std::string_view queries_path,
                                         std::string_view neighbors_path) noexcept {
        mapped_dataset dataset;
        auto [base, base_status] = tools::mapped_rows_t::open(vectors_path);
        if (failed(base_status)) return {std::move(dataset), base_status};
        dataset.base = std::move(base);
        if (!queries_path.empty()) {
            auto [queries, queries_status] = tools::mapped_rows_t::open(queries_path);
            if (failed(queries_status)) return {std::move(dataset), queries_status};
            if (queries.columns() != dataset.base.columns()) return {std::move(dataset), status_t::config_mismatch_k};
            dataset.query_rows = std::move(queries);
        }
        if (!neighbors_path.empty()) {
            auto [neighbors, neighbors_status] = tools::mapped_rows_t::open(neighbors_path);
            if (failed(neighbors_status)) return {std::move(dataset), neighbors_status};
            if (dataset.query_rows && neighbors.rows() != dataset.query_rows.rows())
                return {std::move(dataset), status_t::config_mismatch_k};
            dataset.neighbor_rows = std::move(neighbors);
        }
        return {std::move(dataset), status_t::success_k};
    }

    /** Base rows `[first_row, first_row + rows)`, viewed in place when one part holds them all;
     *  empty when they straddle two parts or run past the last. */
    nk::matrix_view<scalar_type_> vectors(std::size_t first_row, std::size_t rows) const noexcept {
        usearch_rows_view_t const viewed = base.view(first_row, rows);
        if (!viewed.data) return {};
        return {static_cast<scalar_type_ const*>(viewed.data), rows, base.columns()};
    }
    nk::matrix_view<scalar_type_> queries() const noexcept {
        usearch_rows_view_t const viewed = query_rows.view(0, query_rows.rows());
        return {static_cast<scalar_type_ const*>(viewed.data), query_rows.rows(), query_rows.columns()};
    }
    nk::matrix_view<u32_t> neighbors() const noexcept {
        usearch_rows_view_t const viewed = neighbor_rows.view(0, neighbor_rows.rows());
        return {static_cast<u32_t const*>(viewed.data), neighbor_rows.rows(), neighbor_rows.columns()};
    }

    bool has_queries() const noexcept { return static_cast<bool>(query_rows); }
    bool has_ground_truth() const noexcept { return static_cast<bool>(neighbor_rows); }
    std::size_t rows() const noexcept { return base.rows(); }
    std::size_t dimensions() const noexcept { return base.columns(); }
};

/** Bundles the launch state a CPU cascade threads through its compile-time hops: the dataset, the parsed CLI, and the
 *  machine whose pool or topology the run spawns its threads on. */
template <typename dataset_scalar_type_>
struct bench_launch {
    mapped_dataset<dataset_scalar_type_> const& dataset;
    settings_t const& settings;
    machine_t const& machine;
};

/** Opens the RetriEval matrix @p path, or the parts its glob pattern names, in the scalar its
 *  extension names, the hop lowering the file's dtype to the C++ wrapper a run is instantiated
 *  over, with the rows of @p queries_path and @p neighbors_path when named, and exits with what
 *  @p on_dataset makes of it. */
template <typename on_dataset_type_>
int open_dataset(std::string_view path, std::string_view queries_path, std::string_view neighbors_path,
                 on_dataset_type_&& on_dataset) {
    return dispatch_value<nk_f32_k, nk_bf16_k, nk_f16_k, nk_i8_k, nk_u8_k, nk_u1_k>(
        dtype_of_extension(path),
        [&](auto dtype) {
            auto [dataset, status] = mapped_dataset<typename nk::type_for<dtype.value>::type>::open(path, queries_path,
                                                                                                    neighbors_path);
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
            settings.threads.value = max_of<std::size_t>(1, local_cores * 7 / 8);
        }
        spawned = fu_pool_spawn_near_memory_domain(machine.topology, machine.pool, settings.memory_domains.front(),
                                                   settings.threads.value, fu_caller_inclusive_k) == fu_success_k;
    }
    else {
        if (!settings.threads.value) settings.threads.value = max_of<std::size_t>(1, machine.logical_cores * 7 / 8);
        spawned = fu_pool_spawn(machine.topology, machine.pool, settings.threads.value, fu_caller_inclusive_k) ==
                  fu_success_k;
    }
    if (!spawned) {
        fmt::println("Failed to spawn {} threads", settings.threads.value);
        return 1;
    }
    settings.cores = settings.threads.value;
    print(machine);
    print(settings);
    if (!tools::huge_pages_provisioned(settings.huge_pages.value)) return 1;

    int const exit_code = open_dataset(
        settings.base_vectors_path, settings.query_vectors_path, settings.query_neighbors_path,
        [&]<typename dataset_scalar_type_>(mapped_dataset<dataset_scalar_type_> const& dataset) {
            return on_launch(bench_launch<dataset_scalar_type_> {dataset, settings, machine});
        });
    fu_pool_delete(machine.pool);
    fu_fabric_delete(machine.fabric);
    fu_topology_delete(machine.topology);
    return exit_code;
}

#pragma endregion

} // namespace unum::usearch::bench
