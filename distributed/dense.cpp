/**
 *  @file distributed/dense.cpp
 *  @author Ash Vardanian
 *  @date October 12, 2026
 *  @brief Builds one dense index across an MPI world: a shard per rank, committed as a SafeGraphs
 *      manifest.
 *
 *  Every rank indexes a contiguous slice of one RetriEval-layout file on its own GPU or memory
 *  domain, and saves it as one monolith under the name a sharded save gives its parts. Once every
 *  rank reports its part complete, rank 0 writes the manifest beside them: the commit record, so a
 *  run that fails anywhere leaves parts and no checkpoint. Keys come row-aligned from @c --keys,
 *  or are the global row numbers.
 *
 *  The library never sees MPI, and this driver calls both through their C interfaces. Ranks
 *  agree on their slices from the world size alone, and talk only twice: once to stop together
 *  when any of them cannot start, and once to gather the parts.
 */
#include <cstdint> // `std::uint64_t`, `std::int32_t`
#include <cstdio>  // `std::fopen`, `std::fwrite`
#include <cstdlib> // `EXIT_SUCCESS`, `std::getenv`

#include <fcntl.h>    // `open`, `O_RDONLY`
#include <sys/stat.h> // `fstat`, `stat`
#include <unistd.h>   // `pread`, `close`, `fsync`

#include <algorithm>   // `std::min`
#include <array>       // `std::array`
#include <iterator>    // `std::back_inserter`
#include <span>        // `std::span`
#include <string>      // `std::string`
#include <string_view> // `std::string_view`
#include <vector>      // `std::vector`

#include <fmt/format.h>
#include <mpi.h>

#include <forkunion.h> // `fu_topology_new`, `fu_memory_domain_id_at_index`

#include "usearch/dense_index.h"

#include "arguments.hpp" // `bench::parse_arguments`, `bench::print_usage`

namespace unum::usearch::distributed {

using bench::count_t;

/** Two little-endian `u32` counters, rows then columns, open every RetriEval file. */
static constexpr std::size_t dense_header_bytes_k = 8;
static constexpr std::string_view dense_manifest_suffix_k = ".safegraphs.json";

/** Every setting the command line takes, each spelled as its flag with dashes. */
struct dense_settings_t {
    std::string_view vectors;
    std::string_view keys;
    std::string_view output;
    std::string_view stored_dtype;
    std::string_view metric = "angular";
    std::string_view residency;
    count_t connectivity {.minimum = 0};
    count_t expansion_add {.minimum = 0};
    count_t expansion_refine {.minimum = 0};
    count_t max_refines {.minimum = 0};
    count_t batch {.value = 65536};
    count_t threads {.minimum = 0};
};

/** What this rank opened and built, released whichever way the build ends. */
struct dense_rank_t {
    int descriptor = -1;
    int keys_descriptor = -1;
    std::size_t key_bytes = 0;
    std::size_t rows = 0;
    std::size_t columns = 0;
    std::size_t row_bytes = 0;
    usearch_scalar_t scalar = usearch_scalar_unknown_k;
    std::uint64_t first_row = 0;
    std::size_t slice_rows = 0;
    usearch_residency_t residency = usearch_residency_ram_k;
    std::int32_t placement = -1;
    usearch_executor_t executor = nullptr;
    usearch_dense_index_t index = nullptr;

    dense_rank_t() = default;
    dense_rank_t(dense_rank_t const&) = delete;
    dense_rank_t& operator=(dense_rank_t const&) = delete;
    ~dense_rank_t() noexcept {
        usearch_dense_free(index);
        usearch_executor_free(executor);
        if (descriptor >= 0) close(descriptor);
        if (keys_descriptor >= 0) close(keys_descriptor);
    }
};

/** One of two alternating staging buffers: the file fills one while the index ingests the other. */
struct dense_staging_t {
    std::vector<std::uint8_t> rows;
    std::vector<std::uint8_t> key_integers;
    std::vector<usearch_key_t> keys;
    usearch_outcome_t outcome {};

    dense_staging_t() = default;
    dense_staging_t(dense_staging_t const&) = delete;
    dense_staging_t& operator=(dense_staging_t const&) = delete;
    ~dense_staging_t() noexcept { usearch_outcome_wait(&outcome); }
};

/** One saved part as rank 0 gathers it into the manifest. */
struct dense_part_t {
    std::uint64_t bytes = 0;
    std::uint64_t nodes = 0;
};

static usearch_bytes_view_t dense_bytes(std::string_view text) noexcept {
    return {reinterpret_cast<std::uint8_t const*>(text.data()), text.size()};
}

/** The scalar a RetriEval extension names, or @c usearch_scalar_unknown_k. */
static usearch_scalar_t dense_scalar_of_extension(std::string_view path) noexcept {
    if (path.ends_with(".fbin")) return usearch_scalar_f32_k;
    if (path.ends_with(".bf16bin")) return usearch_scalar_bf16_k;
    if (path.ends_with(".f16bin")) return usearch_scalar_f16_k;
    if (path.ends_with(".i8bin")) return usearch_scalar_i8_k;
    if (path.ends_with(".u8bin")) return usearch_scalar_u8_k;
    if (path.ends_with(".u1x8bin")) return usearch_scalar_u1x8_k;
    return usearch_scalar_unknown_k;
}

/** Bytes one row of @p columns logical values takes in a RetriEval file of @p scalar. */
static std::size_t dense_row_bytes(usearch_scalar_t scalar, std::size_t columns) noexcept {
    switch (scalar) {
    case usearch_scalar_f32_k: return columns * 4;
    case usearch_scalar_bf16_k:
    case usearch_scalar_f16_k: return columns * 2;
    case usearch_scalar_u1x8_k: return (columns + 7) / 8;
    default: return columns;
    }
}

/** Reads exactly @p length bytes at @p offset, retrying short reads; false on an error or at the
 *  end of the file. */
static bool dense_read_at(int descriptor, void* buffer, std::size_t length, std::uint64_t offset) noexcept {
    auto* cursor = static_cast<std::uint8_t*>(buffer);
    while (length) {
        ssize_t const read_bytes = pread(descriptor, cursor, length, static_cast<off_t>(offset));
        if (read_bytes <= 0) return false;
        cursor += read_bytes, length -= static_cast<std::size_t>(read_bytes), offset += read_bytes;
    }
    return true;
}

/** The 128-bit key of global row @p row: the integer in the low bytes, the rest zeroed. */
static usearch_key_t dense_key_of(std::uint64_t row) noexcept {
    usearch_key_t key {};
    for (unsigned byte_index = 0; byte_index != 8; ++byte_index)
        key.bytes[15 - byte_index] = static_cast<std::uint8_t>(row >> (byte_index * 8));
    return key;
}

/** Prints a failed call's status and message under the rank that saw it; true when it failed. */
static bool dense_failed(usearch_outcome_t const& outcome, int rank, std::string_view verb) {
    if (outcome.status == usearch_status_success_k) return false;
    fmt::println(stderr, "rank {}: {} failed with {}: {}", rank, verb, usearch_status_name(outcome.status),
                 outcome.message ? outcome.message : "no message");
    return true;
}

/** `<stem>-00001-of-00004.safegraphs` for @p part counted from zero. */
static std::string dense_part_path(std::string_view stem, std::size_t part, std::size_t parts) {
    return fmt::format("{}-{:05}-of-{:05}.safegraphs", stem, part + 1, parts);
}

/** Opens @p path and checks its counters against its size. */
static bool dense_open_vectors(std::string_view path, dense_rank_t& state, int rank) {
    state.scalar = dense_scalar_of_extension(path);
    if (state.scalar == usearch_scalar_unknown_k) {
        fmt::println(stderr, "rank {}: {} is none of .fbin, .bf16bin, .f16bin, .i8bin, .u8bin or .u1x8bin", rank, path);
        return false;
    }
    // Text flags view the tail of their `argv` entry, so the path is null-terminated.
    state.descriptor = open(path.data(), O_RDONLY);
    std::array<std::uint32_t, 2> counters {};
    struct stat file_stat;
    if (state.descriptor < 0 || fstat(state.descriptor, &file_stat) != 0 ||
        !dense_read_at(state.descriptor, counters.data(), sizeof(counters), 0)) {
        fmt::println(stderr, "rank {}: cannot read {}", rank, path);
        return false;
    }
    state.rows = counters[0], state.columns = counters[1];
    state.row_bytes = dense_row_bytes(state.scalar, state.columns);
    if (!state.rows || !state.columns ||
        static_cast<std::uint64_t>(file_stat.st_size) <
            dense_header_bytes_k + static_cast<std::uint64_t>(state.rows) * state.row_bytes) {
        fmt::println(stderr, "rank {}: {} holds {} rows of {} columns, which its {} bytes do not", rank, path,
                     state.rows, state.columns, static_cast<long long>(file_stat.st_size));
        return false;
    }
    return true;
}

/** Opens @p path as the keys of the vectors' rows: a `.u64bin` or `.u128bin` of one little-endian
 *  integer per row, as every RetriEval scalar is stored, landing in a key's low bytes as row
 *  numbers do; a 128-bit integer fills the whole key, so a UUID's integer form keys its row. */
static bool dense_open_keys(std::string_view path, dense_rank_t& state, int rank) {
    state.key_bytes = path.ends_with(".u64bin") ? 8 : path.ends_with(".u128bin") ? 16 : 0;
    if (!state.key_bytes) {
        fmt::println(stderr, "rank {}: --keys={} is neither .u64bin nor .u128bin", rank, path);
        return false;
    }
    // Text flags view the tail of their `argv` entry, so the path is null-terminated.
    state.keys_descriptor = open(path.data(), O_RDONLY);
    std::array<std::uint32_t, 2> counters {};
    struct stat file_stat;
    if (state.keys_descriptor < 0 || fstat(state.keys_descriptor, &file_stat) != 0 ||
        !dense_read_at(state.keys_descriptor, counters.data(), sizeof(counters), 0)) {
        fmt::println(stderr, "rank {}: cannot read {}", rank, path);
        return false;
    }
    if (counters[0] != state.rows || counters[1] != 1 ||
        static_cast<std::uint64_t>(file_stat.st_size) <
            dense_header_bytes_k + static_cast<std::uint64_t>(state.rows) * state.key_bytes) {
        fmt::println(stderr, "rank {}: {} holds {} rows of {} columns, not one key for each of {} rows", rank, path,
                     counters[0], counters[1], state.rows);
        return false;
    }
    return true;
}

/** The device ordinal this rank builds on: the entry of `USEARCH_DEVICES` at the node-local rank,
 *  wrapping around, or that position among every listed device when it is unset. */
static bool dense_device_of(int local_rank, std::size_t devices_count, dense_rank_t& state, int rank) {
    char const* listed = std::getenv("USEARCH_DEVICES");
    if (!listed || !*listed) {
        state.placement = static_cast<std::int32_t>(static_cast<std::size_t>(local_rank) % devices_count);
        return true;
    }
    bench::counts_t ordinals {.minimum = 0, .maximum = devices_count - 1};
    if (bench::parse_counts(listed, ordinals) != bench::argument_status_t::parsed_k) {
        fmt::println(stderr, "rank {}: USEARCH_DEVICES=\"{}\" names no device among the {} listed", rank, listed,
                     devices_count);
        return false;
    }
    state.placement = static_cast<std::int32_t>(ordinals.values[static_cast<std::size_t>(local_rank) % ordinals.size]);
    return true;
}

/** The memory domain this rank builds in: the node-local rank's position among the allocatable
 *  domains, wrapping around, so ranks sharing a node spread over its sockets. */
static bool dense_memory_domain_of(int local_rank, dense_rank_t& state, int rank) {
    fu_topology_t topology = nullptr;
    std::size_t domains_count = 0;
    fu_memory_domain_id_t found = -1;
    bool const listed = fu_topology_new(&topology) == fu_success_k &&
                        fu_memory_domains_count(topology, &domains_count) == fu_success_k && domains_count &&
                        fu_memory_domain_id_at_index(topology, static_cast<std::size_t>(local_rank) % domains_count,
                                                     &found) == fu_success_k;
    if (topology) fu_topology_delete(topology);
    if (!listed) fmt::println(stderr, "rank {}: cannot list the memory domains", rank);
    state.placement = static_cast<std::int32_t>(found);
    return listed;
}

/** Opens the file, takes this rank's slice and creates its executor and empty index, stopping at
 *  the first failure. */
static bool dense_setup(dense_settings_t const& settings, usearch_dense_metric_t metric, int rank, int ranks,
                        int local_rank, dense_rank_t& state) {
    if (!dense_open_vectors(settings.vectors, state, rank)) return false;
    if (!settings.keys.empty() && !dense_open_keys(settings.keys, state, rank)) return false;

    // Ranks share the rows as evenly as integers allow, so every slice is known without asking.
    state.first_row = static_cast<std::uint64_t>(state.rows) * rank / ranks;
    state.slice_rows = static_cast<std::size_t>(static_cast<std::uint64_t>(state.rows) * (rank + 1) / ranks -
                                                state.first_row);
    if (!state.slice_rows) {
        fmt::println(stderr, "rank {}: {} rows leave no slice for each of {} ranks", rank, state.rows, ranks);
        return false;
    }

    usearch_scalar_t quantization = state.scalar;
    if (!settings.stored_dtype.empty()) {
        quantization = usearch_scalar_from_name(dense_bytes(settings.stored_dtype));
        if (quantization == usearch_scalar_unknown_k) {
            fmt::println(stderr, "rank {}: --stored-dtype=\"{}\" names no scalar", rank, settings.stored_dtype);
            return false;
        }
    }

    usearch_outcome_t outcome {};
    std::size_t devices_count = 0;
    usearch_gpu_devices_t devices {.outcome = &outcome, .devices_count = &devices_count};
    usearch_gpu_devices(&devices);
    if (dense_failed(outcome, rank, "listing GPUs")) return false;

    state.residency = !settings.residency.empty()
                          ? usearch_residency_from_name(dense_bytes(settings.residency))
                          : (devices_count ? usearch_residency_vram_k : usearch_residency_ram_k);
    // Unknown spellings read back as ram, so only a round trip proves the name.
    bool const named = settings.residency.empty() || usearch_residency_name(state.residency) == settings.residency;
    if (!named || state.residency == usearch_residency_nvme_k ||
        (state.residency == usearch_residency_vram_k && !devices_count)) {
        fmt::println(stderr, "rank {}: --residency=\"{}\" is neither ram nor vram over a listed GPU", rank,
                     settings.residency.empty() ? "vram" : settings.residency);
        return false;
    }

    if (state.residency == usearch_residency_vram_k) {
        if (!dense_device_of(local_rank, devices_count, state, rank)) return false;
        usearch_vram_options_t const vram {&state.placement, 1};
        usearch_executor_for_vram_t executor {.executor = &state.executor, .outcome = &outcome, .vram = vram};
        usearch_executor_for_vram(&executor);
        if (dense_failed(outcome, rank, "creating the GPU executor")) return false;
        usearch_dense_init_vram_t init {
            .index = &state.index,
            .outcome = &outcome,
            .metric = metric,
            .quantization = quantization,
            .dimensions = state.columns,
            .connectivity = settings.connectivity.value,
            .vram = vram,
        };
        usearch_dense_init_vram(&init);
    }
    else {
        if (!dense_memory_domain_of(local_rank, state, rank)) return false;
        usearch_ram_options_t const ram {&state.placement, 1};
        usearch_executor_for_ram_t executor {.executor = &state.executor,
                                             .outcome = &outcome,
                                             .ram = ram,
                                             .threads = settings.threads.value};
        usearch_executor_for_ram(&executor);
        if (dense_failed(outcome, rank, "creating the CPU executor")) return false;
        usearch_dense_init_ram_t init {
            .index = &state.index,
            .outcome = &outcome,
            .metric = metric,
            .quantization = quantization,
            .dimensions = state.columns,
            .connectivity = settings.connectivity.value,
            .ram = ram,
        };
        usearch_dense_init_ram(&init);
    }
    if (dense_failed(outcome, rank, "creating the index")) return false;

    usearch_dense_reserve_t reserve {.index = state.index,
                                     .executor = state.executor,
                                     .outcome = &outcome,
                                     .capacity = state.slice_rows};
    usearch_dense_reserve(&reserve);
    return !dense_failed(outcome, rank, "reserving");
}

/** Inserts the slice, reading one staging while the index ingests the other. */
static bool dense_insert_slice(dense_settings_t const& settings, dense_rank_t const& state, int rank) {
    std::size_t const batch_rows = std::min(settings.batch.value, state.slice_rows);
    std::array<dense_staging_t, 2> stagings;
    for (dense_staging_t& staging : stagings) {
        staging.rows.resize(batch_rows * state.row_bytes);
        staging.keys.resize(batch_rows);
        staging.key_integers.resize(batch_rows * state.key_bytes);
    }

    for (std::size_t offset = 0, turn = 0; offset < state.slice_rows; offset += batch_rows, turn ^= 1) {
        dense_staging_t& staging = stagings[turn];
        // A zeroed outcome already reads as final, so a staging's first turn waits on nothing.
        if (usearch_outcome_wait(&staging.outcome) != usearch_status_success_k) break;
        std::size_t const count = std::min(batch_rows, state.slice_rows - offset);
        std::uint64_t const row = state.first_row + offset;
        if (!dense_read_at(state.descriptor, staging.rows.data(), count * state.row_bytes,
                           dense_header_bytes_k + row * state.row_bytes)) {
            fmt::println(stderr, "rank {}: cannot read rows {} to {}", rank, row, row + count);
            return false;
        }
        if (state.keys_descriptor >= 0) {
            if (!dense_read_at(state.keys_descriptor, staging.key_integers.data(), count * state.key_bytes,
                               dense_header_bytes_k + row * state.key_bytes)) {
                fmt::println(stderr, "rank {}: cannot read keys {} to {}", rank, row, row + count);
                return false;
            }
            // Little-endian integers into keys stored most significant byte first.
            for (std::size_t position = 0; position != count; ++position) {
                usearch_key_t& key = staging.keys[position];
                key = {};
                for (std::size_t byte_index = 0; byte_index != state.key_bytes; ++byte_index)
                    key.bytes[15 - byte_index] = staging.key_integers[position * state.key_bytes + byte_index];
            }
        }
        else {
            for (std::size_t position = 0; position != count; ++position)
                staging.keys[position] = dense_key_of(row + position);
        }
        usearch_dense_insert_t insert {
            .index = state.index,
            .executor = state.executor,
            .outcome = &staging.outcome,
            .completion = usearch_completion_deferred_k,
            .count = count,
            .keys = {staging.keys.data(), sizeof(usearch_key_t)},
            .vectors = {staging.rows.data(), state.row_bytes, state.columns, state.scalar},
            .expansion_factor = settings.expansion_add.value,
        };
        usearch_dense_insert(&insert);
    }

    bool failed = false;
    for (dense_staging_t& staging : stagings) {
        usearch_outcome_wait(&staging.outcome);
        failed = dense_failed(staging.outcome, rank, "insert") || failed;
    }
    return !failed;
}

/** Inserts, refines and saves this rank's part to @p part_path, measuring it into @p part. */
static bool dense_construct(dense_settings_t const& settings, std::string const& part_path, int rank,
                            dense_rank_t const& state, dense_part_t& part) {
    if (!dense_insert_slice(settings, state, rank)) return false;
    usearch_outcome_t outcome {};
    for (std::size_t pass = 0; pass != settings.max_refines.value; ++pass) {
        std::size_t changed = 0;
        usearch_dense_refine_t refine {
            .index = state.index,
            .executor = state.executor,
            .outcome = &outcome,
            .changed = &changed,
            .expansion_factor = settings.expansion_refine.value,
        };
        usearch_dense_refine(&refine);
        if (dense_failed(outcome, rank, "refining")) return false;
        if (!changed) break;
    }
    usearch_dense_save_t save {.index = state.index,
                               .executor = state.executor,
                               .outcome = &outcome,
                               .path = dense_bytes(part_path)};
    usearch_dense_save(&save);
    if (dense_failed(outcome, rank, "saving")) return false;

    struct stat part_stat;
    if (stat(part_path.c_str(), &part_stat) != 0) {
        fmt::println(stderr, "rank {}: cannot measure {}", rank, part_path);
        return false;
    }
    part = {static_cast<std::uint64_t>(part_stat.st_size), state.slice_rows};
    return true;
}

/** Appends @p text as a JSON string, escaping quotes, backslashes and control bytes. */
static void dense_append_json_string(std::string& json, std::string_view text) {
    json += '"';
    for (unsigned char const character : text) {
        if (character == '"' || character == '\\') json += '\\';
        if (character < 0x20) fmt::format_to(std::back_inserter(json), "\\u{:04x}", character);
        else json += static_cast<char>(character);
    }
    json += '"';
}

/** Commits the checkpoint: the manifest naming every rank's part, its bytes and nodes. */
static bool dense_write_manifest(std::string_view manifest_path, std::string_view part_stem,
                                 std::span<dense_part_t const> parts) {
    std::string json = "{\"files\":[";
    for (std::size_t part = 0; part != parts.size(); ++part) {
        json += part ? ",{\"path\":" : "{\"path\":";
        dense_append_json_string(json, dense_part_path(part_stem, part, parts.size()));
        fmt::format_to(std::back_inserter(json), ",\"bytes\":{},\"nodes\":{}}}", parts[part].bytes, parts[part].nodes);
    }
    json += "]}";
    std::FILE* file = std::fopen(manifest_path.data(), "wb");
    if (!file) return false;
    bool const written = std::fwrite(json.data(), 1, json.size(), file) == json.size() && std::fflush(file) == 0 &&
                         fsync(fileno(file)) == 0;
    return std::fclose(file) == 0 && written;
}

static int dense_build(int rank, int ranks, int local_rank, int argc, char const* const* argv) {
    dense_settings_t settings;
    std::array<bench::flag_t, 12> const flags {{
        {"--vectors", "path",
         "required: RetriEval vectors every rank reads: .fbin, .bf16bin, .f16bin, .i8bin, .u8bin or .u1x8bin",
         &settings.vectors},
        {"--keys", "path", "defaults to row numbers: a .u64bin or .u128bin of one key per vector row", &settings.keys},
        {"--output", "path", "required: the .safegraphs.json manifest to commit, its parts written beside it",
         &settings.output},
        {"--stored-dtype", "name",
         "defaults to the file's: bf16, f16, e5m2, e4m3, e3m2, e2m3, i8, i4x2, u8, u4x2 or u1x8",
         &settings.stored_dtype},
        {"--metric", "name", "defaults to angular: inner, angular, euclidean, hamming or jaccard", &settings.metric},
        {"--residency", "name", "defaults to vram when a GPU is listed, else ram", &settings.residency},
        {"--connectivity", "count", "defaults to the library's derivation: graph degree", &settings.connectivity},
        {"--expansion-add", "count", "defaults to the library's derivation: candidate pool while inserting",
         &settings.expansion_add},
        {"--max-refines", "count", "defaults to 0: refinement passes after ingestion, stopping early once settled",
         &settings.max_refines},
        {"--expansion-refine", "count", "defaults to the library's derivation: candidate pool while refining",
         &settings.expansion_refine},
        {"--batch", "count", "defaults to 65536: rows per insertion, two batches staged at once", &settings.batch},
        {"--threads", "count", "defaults to 7/8 of the domain's cores: executor threads, ram only", &settings.threads},
    }};

    // Every rank parses the same arguments, so they agree on failing, and only rank 0 says why.
    bench::arguments_t const parsed = bench::parse_arguments(argc, argv, flags, {});
    if (parsed.status == bench::argument_status_t::help_k) {
        if (rank == 0) {
            bench::print_usage("usearch_distributed_dense --vectors <path> --output <path>.safegraphs.json [flags]",
                               flags);
            fmt::println("  USEARCH_DEVICES{:19}environment: device ordinals the ranks of a node take in turn", "");
        }
        return EXIT_SUCCESS;
    }
    if (parsed.status != bench::argument_status_t::parsed_k) {
        if (rank == 0 && !parsed.flag) fmt::println(stderr, "{} is not a flag, see --help", parsed.argument);
        else if (rank == 0)
            fmt::println(stderr, "{}=\"{}\" does not parse, expected {}", parsed.flag->name, parsed.value,
                         bench::grammar(*parsed.flag));
        return EXIT_FAILURE;
    }
    if (settings.vectors.empty() || !settings.output.ends_with(dense_manifest_suffix_k)) {
        if (rank == 0)
            fmt::println(stderr, "--vectors and --output ending in {} are required", dense_manifest_suffix_k);
        return EXIT_FAILURE;
    }
    usearch_dense_metric_t const metric = usearch_dense_metric_from_name(dense_bytes(settings.metric));
    if (metric == usearch_dense_metric_unknown_k) {
        if (rank == 0) fmt::println(stderr, "--metric=\"{}\" names no metric", settings.metric);
        return EXIT_FAILURE;
    }

    double const started = MPI_Wtime();
    dense_rank_t state;
    int any_failed = !dense_setup(settings, metric, rank, ranks, local_rank, state);
    // One rank unable to start stops every rank before any of them spends hours building.
    MPI_Allreduce(MPI_IN_PLACE, &any_failed, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);

    std::string_view const stem = settings.output.substr(0, settings.output.size() - dense_manifest_suffix_k.size());
    std::string const part_path = dense_part_path(stem, static_cast<std::size_t>(rank),
                                                  static_cast<std::size_t>(ranks));
    dense_part_t part;
    if (!any_failed) any_failed = !dense_construct(settings, part_path, rank, state, part);
    if (!any_failed)
        fmt::println("rank {}: {} rows on {} {} {} in {:.1f} s", rank, state.slice_rows,
                     usearch_residency_name(state.residency),
                     state.residency == usearch_residency_vram_k ? "device" : "memory domain", state.placement,
                     MPI_Wtime() - started);
    MPI_Allreduce(MPI_IN_PLACE, &any_failed, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
    if (any_failed) return EXIT_FAILURE;

    std::vector<dense_part_t> parts(rank == 0 ? static_cast<std::size_t>(ranks) : 0);
    MPI_Gather(&part, 2, MPI_UINT64_T, parts.data(), 2, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    if (rank != 0) return EXIT_SUCCESS;
    std::size_t const slash = stem.rfind('/');
    std::string_view const part_stem = slash == std::string_view::npos ? stem : stem.substr(slash + 1);
    if (!dense_write_manifest(settings.output, part_stem, parts)) {
        fmt::println(stderr, "rank 0: cannot write {}", settings.output);
        return EXIT_FAILURE;
    }
    fmt::println("committed {} over {} parts of {} rows in {:.1f} s", settings.output, ranks, state.rows,
                 MPI_Wtime() - started);
    return EXIT_SUCCESS;
}

} // namespace unum::usearch::distributed

namespace us = unum::usearch;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, ranks = 1, local_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    MPI_Comm node;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
    MPI_Comm_rank(node, &local_rank);
    MPI_Comm_free(&node);
    int const status = us::distributed::dense_build(rank, ranks, local_rank, argc, argv);
    MPI_Finalize();
    return status;
}
