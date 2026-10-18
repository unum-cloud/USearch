/**
 *  @file tools/dense.cpp
 *  @author Ash Vardanian
 *  @date October 17, 2026
 *  @brief The dense verbs: insert, absorb, refine, split and health, each over SafeGraphs
 *      checkpoints at any address.
 *
 *  @c insert indexes a contiguous slice of one RetriEval file on a GPU or in RAM and saves it,
 *  @c absorb merges checkpoints into one, donor by donor, @c refine re-selects a checkpoint's
 *  neighbors, @c split cuts one into the parts of a manifest, and @c health audits how much of one
 *  a search reaches. Keys come row-aligned from @c --keys, or are the dataset row numbers, so
 *  slices built apart keep distinct keys when merged.
 *
 *  Across several ranks, @c insert is a distributed build: rank r of R takes rows ⌊N·r/R⌋ to
 *  ⌊N·(r+1)/R⌋ of the slice, saves them as one part under the name a sharded save gives it, and
 *  once every rank reports its part complete, rank 0 writes the manifest beside them, the commit
 *  record, so a run failing anywhere leaves parts and no checkpoint. Ranks agree on their slices
 *  from the world size alone, and talk only to stop together and to gather the parts.
 */
#include <cstdint> // `std::uint64_t`, `std::int32_t`
#include <cstdio>  // `std::fopen`, `std::fwrite`
#include <cstdlib> // `EXIT_SUCCESS`, `EXIT_FAILURE`, `std::getenv`

#include <fcntl.h>    // `open`, `O_RDONLY`
#include <sys/stat.h> // `fstat`, `stat`, `S_ISREG`
#include <unistd.h>   // `pread`, `close`, `fsync`

#include <algorithm>   // `std::min`
#include <array>       // `std::array`
#include <chrono>      // `std::chrono::steady_clock`
#include <iterator>    // `std::back_inserter`
#include <span>        // `std::span`
#include <string>      // `std::string`
#include <string_view> // `std::string_view`
#include <vector>      // `std::vector`

#include <fmt/format.h>

#include <forkunion.h> // `fu_topology_new`, `fu_memory_domain_id_at_index`

#include "usearch/dense_index.h"

#include "arguments.hpp" // `bench::count_t`, `bench::parse_counts`
#include "world.hpp"     // `world_t`, `parse_verb`

namespace unum::usearch::tools {

using bench::count_t;
using dense_clock_t = std::chrono::steady_clock;

/** Two little-endian `u32` counters, rows then columns, open every RetriEval file. */
static constexpr std::size_t dense_header_bytes_k = 8;
static constexpr std::string_view dense_manifest_suffix_k = ".safegraphs.json";

/** Every setting the command line takes, each spelled as its flag with dashes. */
struct dense_settings_t {
    std::string_view keys;
    std::string_view save;
    std::string_view stored_dtype;
    std::string_view metric = "angular";
    std::string_view residency;
    count_t first_vector {.minimum = 0};
    count_t rows {.minimum = 0};
    count_t parts {.minimum = 2};
    count_t connectivity {.minimum = 0};
    count_t expansion_add {.minimum = 0};
    count_t expansion_refine {.minimum = 0};
    count_t max_refines {.minimum = 0};
    count_t batch {.value = 65536};
    count_t threads {.minimum = 0};
};

/** The RetriEval file an insert reads and its keys, closed whichever way the insert ends. */
struct dense_source_t {
    int descriptor = -1;
    int keys_descriptor = -1;
    std::size_t key_bytes = 0;
    std::size_t rows = 0;
    std::size_t columns = 0;
    std::size_t row_bytes = 0;
    usearch_scalar_t scalar = usearch_scalar_unknown_k;
    std::uint64_t first_row = 0;
    std::size_t slice_rows = 0;

    dense_source_t() = default;
    dense_source_t(dense_source_t const&) = delete;
    dense_source_t& operator=(dense_source_t const&) = delete;
    ~dense_source_t() noexcept {
        if (descriptor >= 0) close(descriptor);
        if (keys_descriptor >= 0) close(keys_descriptor);
    }
};

/** The executor and the index handles a verb holds, freed whichever way it ends. */
struct dense_engine_t {
    usearch_residency_t residency = usearch_residency_ram_k;
    std::int32_t placement = -1;
    usearch_executor_t executor = nullptr;
    std::vector<usearch_dense_index_t> indexes;

    dense_engine_t() = default;
    dense_engine_t(dense_engine_t const&) = delete;
    dense_engine_t& operator=(dense_engine_t const&) = delete;
    ~dense_engine_t() noexcept {
        for (usearch_dense_index_t index : indexes) usearch_dense_free(index);
        usearch_executor_free(executor);
    }
};

/** One of two alternating staging buffers: the file fills one while the index ingests the other. */
struct dense_staging_t {
    std::vector<usearch_byte_t> rows;
    std::vector<usearch_byte_t> key_integers;
    std::vector<usearch_key_t> keys;
    usearch_outcome_t outcome {};

    dense_staging_t() = default;
    dense_staging_t(dense_staging_t const&) = delete;
    dense_staging_t& operator=(dense_staging_t const&) = delete;
    ~dense_staging_t() noexcept { usearch_outcome_wait(&outcome); }
};

/** One saved part as the manifest records it. */
struct dense_part_t {
    std::uint64_t bytes = 0;
    std::uint64_t nodes = 0;
};

static usearch_bytes_view_t dense_bytes(std::string_view text) noexcept {
    return {reinterpret_cast<usearch_byte_t const*>(text.data()), text.size()};
}

static double dense_seconds_since(dense_clock_t::time_point started) noexcept {
    return std::chrono::duration<double>(dense_clock_t::now() - started).count();
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
    auto* cursor = static_cast<usearch_byte_t*>(buffer);
    while (length) {
        ssize_t const read_bytes = pread(descriptor, cursor, length, static_cast<off_t>(offset));
        if (read_bytes <= 0) return false;
        cursor += read_bytes, length -= static_cast<std::size_t>(read_bytes), offset += read_bytes;
    }
    return true;
}

/** The 128-bit key of dataset row @p row: the integer in the low bytes, the rest zeroed. */
static usearch_key_t dense_key_of(std::uint64_t row) noexcept {
    usearch_key_t key {};
    for (unsigned byte_index = 0; byte_index != 8; ++byte_index)
        key.bytes[15 - byte_index] = static_cast<usearch_byte_t>(row >> (byte_index * 8));
    return key;
}

/** Prints a failed call's status and message under the rank that saw it; true when it failed. */
static bool dense_failed(world_t const& world, usearch_outcome_t const& outcome, std::string_view verb) {
    if (outcome.status == usearch_status_success_k) return false;
    fmt::println(stderr, "{}{} failed with {}: {}", rank_label(world), verb, usearch_status_name(outcome.status),
                 outcome.message ? outcome.message : "no message");
    return true;
}

/** Bytes the regular file at @p path holds, or zero for a device, an SPDK address or a failure. */
static std::uint64_t dense_file_bytes(std::string const& path) noexcept {
    struct stat file_stat;
    bool const measured = stat(path.c_str(), &file_stat) == 0 && S_ISREG(file_stat.st_mode);
    return measured ? static_cast<std::uint64_t>(file_stat.st_size) : 0;
}

/** `<stem>-00001-of-00004.safegraphs` for @p part counted from zero. */
static std::string dense_part_path(std::string_view stem, std::size_t part, std::size_t parts) {
    return fmt::format("{}-{:05}-of-{:05}.safegraphs", stem, part + 1, parts);
}

/** Opens @p path and checks its counters against its size. */
static bool dense_open_vectors(world_t const& world, std::string_view path, dense_source_t& source) {
    source.scalar = dense_scalar_of_extension(path);
    if (source.scalar == usearch_scalar_unknown_k) {
        fmt::println(stderr, "{}{} is none of .fbin, .bf16bin, .f16bin, .i8bin, .u8bin or .u1x8bin", rank_label(world),
                     path);
        return false;
    }
    // Positionals view their whole `argv` entry, so the path is null-terminated.
    source.descriptor = open(path.data(), O_RDONLY);
    std::array<std::uint32_t, 2> counters {};
    struct stat file_stat;
    if (source.descriptor < 0 || fstat(source.descriptor, &file_stat) != 0 ||
        !dense_read_at(source.descriptor, counters.data(), sizeof(counters), 0)) {
        fmt::println(stderr, "{}cannot read {}", rank_label(world), path);
        return false;
    }
    source.rows = counters[0], source.columns = counters[1];
    source.row_bytes = dense_row_bytes(source.scalar, source.columns);
    if (!source.rows || !source.columns ||
        static_cast<std::uint64_t>(file_stat.st_size) <
            dense_header_bytes_k + static_cast<std::uint64_t>(source.rows) * source.row_bytes) {
        fmt::println(stderr, "{}{} holds {} rows of {} columns, which its {} bytes do not", rank_label(world), path,
                     source.rows, source.columns, static_cast<long long>(file_stat.st_size));
        return false;
    }
    return true;
}

/** Opens @p path as the keys of the vectors' rows: a `.u64bin` or `.u128bin` of one little-endian
 *  integer per row, as every RetriEval scalar is stored, landing in a key's low bytes as row
 *  numbers do; a 128-bit integer fills the whole key, so a UUID's integer form keys its row. */
static bool dense_open_keys(world_t const& world, std::string_view path, dense_source_t& source) {
    source.key_bytes = path.ends_with(".u64bin") ? 8 : path.ends_with(".u128bin") ? 16 : 0;
    if (!source.key_bytes) {
        fmt::println(stderr, "{}--keys={} is neither .u64bin nor .u128bin", rank_label(world), path);
        return false;
    }
    // Text flags view the tail of their `argv` entry, so the path is null-terminated.
    source.keys_descriptor = open(path.data(), O_RDONLY);
    std::array<std::uint32_t, 2> counters {};
    struct stat file_stat;
    if (source.keys_descriptor < 0 || fstat(source.keys_descriptor, &file_stat) != 0 ||
        !dense_read_at(source.keys_descriptor, counters.data(), sizeof(counters), 0)) {
        fmt::println(stderr, "{}cannot read {}", rank_label(world), path);
        return false;
    }
    if (counters[0] != source.rows || counters[1] != 1 ||
        static_cast<std::uint64_t>(file_stat.st_size) <
            dense_header_bytes_k + static_cast<std::uint64_t>(source.rows) * source.key_bytes) {
        fmt::println(stderr, "{}{} holds {} rows of {} columns, not one key for each of {} rows", rank_label(world),
                     path, counters[0], counters[1], source.rows);
        return false;
    }
    return true;
}

/** The device this rank builds on: the entry of `USEARCH_DEVICES` at the node-local rank, wrapping
 *  around, or that position among every listed device when it is unset. */
static bool dense_device_of(world_t const& world, std::size_t devices_count, std::int32_t& device) {
    char const* const listed = std::getenv("USEARCH_DEVICES");
    if (!listed || !*listed) {
        device = static_cast<std::int32_t>(world.local_rank % devices_count);
        return true;
    }
    bench::counts_t ordinals {.minimum = 0, .maximum = devices_count - 1};
    if (bench::parse_counts(listed, ordinals) != bench::argument_status_t::parsed_k) {
        fmt::println(stderr, "{}USEARCH_DEVICES=\"{}\" names no device among the {} listed", rank_label(world), listed,
                     devices_count);
        return false;
    }
    device = static_cast<std::int32_t>(ordinals.values[world.local_rank % ordinals.size]);
    return true;
}

/** Creates a CPU executor over the memory domain at the node-local rank's position, wrapping
 *  around, so ranks sharing a node spread over its sockets; every index of the verb lives there
 *  as one shard, so a saved monolith stays one graph. */
static bool dense_engine_for_ram(world_t const& world, count_t const& threads, dense_engine_t& engine) {
    fu_topology_t topology = nullptr;
    std::size_t domains_count = 0;
    fu_memory_domain_id_t found = -1;
    bool const listed = fu_topology_new(&topology) == fu_success_k &&
                        fu_memory_domains_count(topology, &domains_count) == fu_success_k && domains_count &&
                        fu_memory_domain_id_at_index(topology, world.local_rank % domains_count, &found) ==
                            fu_success_k;
    if (topology) fu_topology_delete(topology);
    if (!listed) {
        fmt::println(stderr, "{}cannot list the memory domains", rank_label(world));
        return false;
    }
    engine.residency = usearch_residency_ram_k;
    engine.placement = static_cast<std::int32_t>(found);
    usearch_outcome_t outcome {};
    usearch_executor_for_ram_t executor {.executor = &engine.executor,
                                         .outcome = &outcome,
                                         .ram = {&engine.placement, 1},
                                         .threads = threads.value};
    usearch_executor_for_ram(&executor);
    return !dense_failed(world, outcome, "creating the CPU executor");
}

/** Copies the checkpoint at @p address into @p index, a new RAM handle in the engine's domain. */
static bool dense_load(world_t const& world, dense_engine_t const& engine, std::string_view address,
                       usearch_dense_index_t& index) {
    dense_clock_t::time_point const started = dense_clock_t::now();
    usearch_outcome_t outcome {};
    usearch_dense_load_ram_t load {.index = &index,
                                   .executor = engine.executor,
                                   .outcome = &outcome,
                                   .path = dense_bytes(address),
                                   .ram = {&engine.placement, 1}};
    usearch_dense_load_ram(&load);
    if (dense_failed(world, outcome, fmt::format("loading {}", address))) return false;
    fmt::println("loaded {} in {:.1f} s", address, dense_seconds_since(started));
    return true;
}

static bool dense_describe(world_t const& world, dense_engine_t const& engine, usearch_dense_index_t index,
                           usearch_dense_index_description_t& description) {
    usearch_outcome_t outcome {};
    usearch_dense_describe_t describe {.index = index,
                                       .executor = engine.executor,
                                       .outcome = &outcome,
                                       .description = &description};
    usearch_dense_describe(&describe);
    return !dense_failed(world, outcome, "describing");
}

/** Saves @p index under @p address and prints the step. */
static bool dense_save(world_t const& world, dense_engine_t const& engine, usearch_dense_index_t index,
                       std::string const& address) {
    dense_clock_t::time_point const started = dense_clock_t::now();
    usearch_outcome_t outcome {};
    usearch_dense_save_t save {.index = index,
                               .executor = engine.executor,
                               .outcome = &outcome,
                               .path = dense_bytes(address)};
    usearch_dense_save(&save);
    if (dense_failed(world, outcome, fmt::format("saving {}", address))) return false;
    double const seconds = dense_seconds_since(started);
    if (std::uint64_t const bytes = dense_file_bytes(address))
        fmt::println("{}saved {}: {} bytes in {:.1f} s", rank_label(world), address, bytes, seconds);
    else fmt::println("{}saved {} in {:.1f} s", rank_label(world), address, seconds);
    return true;
}

/** Runs up to @p passes refinements over @p index, stopping once one changes nothing. */
static bool dense_refine_passes(world_t const& world, dense_engine_t const& engine, usearch_dense_index_t index,
                                std::size_t passes, std::size_t expansion_factor) {
    usearch_outcome_t outcome {};
    for (std::size_t pass = 0; pass != passes; ++pass) {
        dense_clock_t::time_point const started = dense_clock_t::now();
        std::size_t changed = 0;
        usearch_reachability_t reachability {};
        usearch_dense_refine_t refine {
            .index = index,
            .executor = engine.executor,
            .outcome = &outcome,
            .changed = &changed,
            .reachability = &reachability,
            .expansion_factor = expansion_factor,
        };
        usearch_dense_refine(&refine);
        if (dense_failed(world, outcome, "refining")) return false;
        fmt::println("{}refined pass {}: {} changed, {} of {} reachable in {:.1f} s", rank_label(world), pass + 1,
                     changed, reachability.reachable, reachability.audited, dense_seconds_since(started));
        if (!changed) break;
    }
    return true;
}

/** Inserts the slice, reading one staging while the index ingests the other. */
static bool dense_insert_slice(world_t const& world, dense_settings_t const& settings, dense_source_t const& source,
                               dense_engine_t const& engine) {
    std::size_t const batch_rows = std::min(settings.batch.value, source.slice_rows);
    std::array<dense_staging_t, 2> stagings;
    for (dense_staging_t& staging : stagings) {
        staging.rows.resize(batch_rows * source.row_bytes);
        staging.keys.resize(batch_rows);
        staging.key_integers.resize(batch_rows * source.key_bytes);
    }

    for (std::size_t offset = 0, turn = 0; offset < source.slice_rows; offset += batch_rows, turn ^= 1) {
        dense_staging_t& staging = stagings[turn];
        // A zeroed outcome already reads as final, so a staging's first turn waits on nothing.
        if (usearch_outcome_wait(&staging.outcome) != usearch_status_success_k) break;
        std::size_t const count = std::min(batch_rows, source.slice_rows - offset);
        std::uint64_t const row = source.first_row + offset;
        if (!dense_read_at(source.descriptor, staging.rows.data(), count * source.row_bytes,
                           dense_header_bytes_k + row * source.row_bytes)) {
            fmt::println(stderr, "{}cannot read rows {} to {}", rank_label(world), row, row + count);
            return false;
        }
        if (source.keys_descriptor >= 0) {
            if (!dense_read_at(source.keys_descriptor, staging.key_integers.data(), count * source.key_bytes,
                               dense_header_bytes_k + row * source.key_bytes)) {
                fmt::println(stderr, "{}cannot read keys {} to {}", rank_label(world), row, row + count);
                return false;
            }
            // Little-endian integers into keys stored most significant byte first.
            for (std::size_t position = 0; position != count; ++position) {
                usearch_key_t& key = staging.keys[position];
                key = {};
                for (std::size_t byte_index = 0; byte_index != source.key_bytes; ++byte_index)
                    key.bytes[15 - byte_index] = staging.key_integers[position * source.key_bytes + byte_index];
            }
        }
        else {
            for (std::size_t position = 0; position != count; ++position)
                staging.keys[position] = dense_key_of(row + position);
        }
        usearch_dense_insert_t insert {
            .index = engine.indexes[0],
            .executor = engine.executor,
            .outcome = &staging.outcome,
            .completion = usearch_completion_deferred_k,
            .count = count,
            .keys = {staging.keys.data(), sizeof(usearch_key_t)},
            .vectors = {staging.rows.data(), source.row_bytes, source.columns, source.scalar},
            .expansion_factor = settings.expansion_add.value,
        };
        usearch_dense_insert(&insert);
    }

    bool failed = false;
    for (dense_staging_t& staging : stagings) {
        usearch_outcome_wait(&staging.outcome);
        failed = dense_failed(world, staging.outcome, "insert") || failed;
    }
    return !failed;
}

/** Opens the vectors, takes this rank's share of the slice, creates the executor and the empty
 *  index on the residency asked for, and reserves the share's rows, stopping at the first
 *  failure. */
static bool dense_insert_setup(world_t const& world, dense_settings_t const& settings, std::string_view vectors,
                               usearch_dense_metric_t metric, dense_source_t& source, dense_engine_t& engine) {
    std::string const rank = rank_label(world);
    if (!dense_open_vectors(world, vectors, source)) return false;
    if (!settings.keys.empty() && !dense_open_keys(world, settings.keys, source)) return false;
    std::uint64_t const first = settings.first_vector.value;
    if (first >= source.rows || settings.rows.value > source.rows - first) {
        fmt::println(stderr, "{}--first-vector={} --rows={} reach past the {} rows of {}", rank, first,
                     settings.rows.value, source.rows, vectors);
        return false;
    }
    // Ranks share the slice as evenly as integers allow, so every share is known without asking.
    std::uint64_t const slice = settings.rows.value ? settings.rows.value : source.rows - first;
    source.first_row = first + slice * world.rank / world.size;
    source.slice_rows = static_cast<std::size_t>(first + slice * (world.rank + 1) / world.size - source.first_row);
    if (!source.slice_rows) {
        fmt::println(stderr, "{}{} rows leave no share for each of {} ranks", rank, slice, world.size);
        return false;
    }

    usearch_scalar_t quantization = source.scalar;
    if (!settings.stored_dtype.empty()) {
        quantization = usearch_scalar_from_name(dense_bytes(settings.stored_dtype));
        if (quantization == usearch_scalar_unknown_k) {
            fmt::println(stderr, "{}--stored-dtype=\"{}\" names no scalar", rank, settings.stored_dtype);
            return false;
        }
    }

    usearch_outcome_t outcome {};
    std::size_t devices_count = 0;
    usearch_gpu_devices_t devices {.outcome = &outcome, .devices_count = &devices_count};
    usearch_gpu_devices(&devices);
    if (dense_failed(world, outcome, "listing GPUs")) return false;

    engine.residency = !settings.residency.empty()
                           ? usearch_residency_from_name(dense_bytes(settings.residency))
                           : (devices_count ? usearch_residency_vram_k : usearch_residency_ram_k);
    // Unknown spellings read back as ram, so only a round trip proves the name.
    bool const named = settings.residency.empty() || usearch_residency_name(engine.residency) == settings.residency;
    if (!named || engine.residency == usearch_residency_nvme_k ||
        (engine.residency == usearch_residency_vram_k && !devices_count)) {
        fmt::println(stderr, "{}--residency=\"{}\" is neither ram nor vram over a listed GPU", rank,
                     settings.residency.empty() ? "vram" : settings.residency);
        return false;
    }

    engine.indexes.resize(1);
    if (engine.residency == usearch_residency_vram_k) {
        if (!dense_device_of(world, devices_count, engine.placement)) return false;
        usearch_vram_options_t const vram {&engine.placement, 1};
        usearch_executor_for_vram_t executor {.executor = &engine.executor, .outcome = &outcome, .vram = vram};
        usearch_executor_for_vram(&executor);
        if (dense_failed(world, outcome, "creating the GPU executor")) return false;
        usearch_dense_init_vram_t init {
            .index = &engine.indexes[0],
            .outcome = &outcome,
            .metric = metric,
            .quantization = quantization,
            .dimensions = source.columns,
            .connectivity = settings.connectivity.value,
            .vram = vram,
        };
        usearch_dense_init_vram(&init);
    }
    else {
        if (!dense_engine_for_ram(world, settings.threads, engine)) return false;
        usearch_dense_init_ram_t init {
            .index = &engine.indexes[0],
            .outcome = &outcome,
            .metric = metric,
            .quantization = quantization,
            .dimensions = source.columns,
            .connectivity = settings.connectivity.value,
            .ram = {&engine.placement, 1},
        };
        usearch_dense_init_ram(&init);
    }
    if (dense_failed(world, outcome, "creating the index")) return false;

    usearch_dense_reserve_t reserve {.index = engine.indexes[0],
                                     .executor = engine.executor,
                                     .outcome = &outcome,
                                     .capacity = source.slice_rows};
    usearch_dense_reserve(&reserve);
    return !dense_failed(world, outcome, "reserving");
}

/** Inserts this rank's share, refines it and saves it to @p address, printing each step. */
static bool dense_construct(world_t const& world, dense_settings_t const& settings, dense_source_t const& source,
                            dense_engine_t const& engine, std::string const& address,
                            dense_clock_t::time_point started) {
    if (!dense_insert_slice(world, settings, source, engine)) return false;
    fmt::println("{}inserted rows {} to {} on {} {} {} in {:.1f} s", rank_label(world), source.first_row,
                 source.first_row + source.slice_rows, usearch_residency_name(engine.residency),
                 engine.residency == usearch_residency_vram_k ? "device" : "memory domain", engine.placement,
                 dense_seconds_since(started));
    return dense_refine_passes(world, engine, engine.indexes[0], settings.max_refines.value,
                               settings.expansion_refine.value) &&
           dense_save(world, engine, engine.indexes[0], address);
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

/** Commits the checkpoint: the manifest naming every part, its bytes and nodes. */
static bool dense_write_manifest(std::string_view manifest_path, std::span<dense_part_t const> parts) {
    std::string_view const stem = manifest_path.substr(0, manifest_path.size() - dense_manifest_suffix_k.size());
    std::size_t const slash = stem.rfind('/');
    std::string_view const part_stem = slash == std::string_view::npos ? stem : stem.substr(slash + 1);
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

int dense_insert(world_t& world, int argc, char const* const* argv) {
    dense_settings_t settings;
    std::array<bench::flag_t, 13> const flags {{
        {"--save", "address", "required: where the index is saved; several ranks save a .safegraphs.json manifest",
         &settings.save},
        {"--first-vector", "row", "defaults to 0: the first row of the slice to index", &settings.first_vector},
        {"--rows", "count", "defaults to every row past --first-vector: rows in the slice", &settings.rows},
        {"--keys", "path", "defaults to row numbers: a .u64bin or .u128bin of one key per vector row", &settings.keys},
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
    std::array<std::string_view, 1> positionals;
    std::size_t given = 0;
    if (std::optional<int> const exit = parse_verb(
            world, argc, argv, "usearch dense insert <vectors> --save <address> [flags]", flags, positionals, given))
        return *exit;
    bool const parted = world.size > 1;
    bool const speaks = world.rank == 0;
    if (given != 1 || settings.save.empty() || (parted && !settings.save.ends_with(dense_manifest_suffix_k))) {
        if (speaks)
            fmt::println(stderr, "the vectors and --save are required, and across ranks --save ends in {}",
                         dense_manifest_suffix_k);
        return EXIT_FAILURE;
    }
    usearch_dense_metric_t const metric = usearch_dense_metric_from_name(dense_bytes(settings.metric));
    if (metric == usearch_dense_metric_unknown_k) {
        if (speaks) fmt::println(stderr, "--metric=\"{}\" names no metric", settings.metric);
        return EXIT_FAILURE;
    }
    if (!parted && !checkpoints_admit_save(settings.save)) return EXIT_FAILURE;

    dense_clock_t::time_point const started = dense_clock_t::now();
    dense_source_t source;
    dense_engine_t engine;
    // One rank unable to start stops every rank before any of them spends hours building.
    if (world.barrier(!dense_insert_setup(world, settings, positionals[0], metric, source, engine)))
        return EXIT_FAILURE;

    std::string_view const stem = settings.save.substr(0, settings.save.size() - dense_manifest_suffix_k.size());
    std::string const address = parted ? dense_part_path(stem, world.rank, world.size) : std::string(settings.save);
    bool const built = dense_construct(world, settings, source, engine, address, started);
    if (world.barrier(!built)) return EXIT_FAILURE;
    if (!parted) return EXIT_SUCCESS;

    std::array<std::uint64_t, 2> const sent {dense_file_bytes(address), source.slice_rows};
    std::vector<std::uint64_t> gathered(speaks ? sent.size() * world.size : 0);
    world.gather(sent, gathered);
    if (!speaks) return EXIT_SUCCESS;
    std::vector<dense_part_t> parts(world.size);
    for (std::size_t part = 0; part != parts.size(); ++part)
        parts[part] = {gathered[part * sent.size()], gathered[part * sent.size() + 1]};
    if (!dense_write_manifest(settings.save, parts)) {
        fmt::println(stderr, "rank 0: cannot write {}", settings.save);
        return EXIT_FAILURE;
    }
    fmt::println("committed {} over {} parts of {} rows in {:.1f} s", settings.save, world.size,
                 settings.rows.value ? settings.rows.value : source.rows - settings.first_vector.value,
                 dense_seconds_since(started));
    return EXIT_SUCCESS;
}

int dense_absorb(world_t& world, int argc, char const* const* argv) {
    dense_settings_t settings;
    std::array<bench::flag_t, 3> const flags {{
        {"--save", "address", "required: where the merged index is saved", &settings.save},
        {"--expansion-refine", "count", "defaults to the library's derivation: candidate pool while refining",
         &settings.expansion_refine},
        {"--threads", "count", "defaults to 7/8 of the domain's cores: executor threads", &settings.threads},
    }};
    // Every argument is at most one positional, so the arguments bound the donors.
    std::vector<std::string_view> positionals(static_cast<std::size_t>(argc));
    std::size_t given = 0;
    if (std::optional<int> const exit = parse_verb(world, argc, argv,
                                                   "usearch dense absorb <base> <donors>... --save <address> [flags]",
                                                   flags, positionals, given))
        return *exit;
    if (!runs_alone(world)) return EXIT_FAILURE;
    if (given < 2 || settings.save.empty()) {
        fmt::println(stderr, "a base, at least one donor and --save are required");
        return EXIT_FAILURE;
    }
    if (!checkpoints_admit_save(settings.save)) return EXIT_FAILURE;

    dense_engine_t engine;
    if (!dense_engine_for_ram(world, settings.threads, engine)) return EXIT_FAILURE;
    engine.indexes.resize(2);
    if (!dense_load(world, engine, positionals[0], engine.indexes[0])) return EXIT_FAILURE;
    for (std::string_view const donor : std::span {positionals}.subspan(1, given - 1)) {
        usearch_dense_index_description_t description {};
        if (!dense_load(world, engine, donor, engine.indexes[1])) return EXIT_FAILURE;
        if (!dense_describe(world, engine, engine.indexes[1], description)) return EXIT_FAILURE;

        dense_clock_t::time_point const started = dense_clock_t::now();
        usearch_outcome_t outcome {};
        std::size_t changed = 0;
        usearch_reachability_t reachability {};
        // A consumed donor's slot is nulled; a refused one stays for the engine to free.
        usearch_dense_absorb_t absorb {
            .index = engine.indexes[0],
            .executor = engine.executor,
            .outcome = &outcome,
            .count = 1,
            .donors = &engine.indexes[1],
            .changed = &changed,
            .reachability = &reachability,
            .expansion_factor = settings.expansion_refine.value,
        };
        usearch_dense_absorb(&absorb);
        if (dense_failed(world, outcome, fmt::format("absorbing {}", donor))) return EXIT_FAILURE;
        fmt::println("absorbed {}: {} rows in {:.1f} s, {} changed, {} of {} reachable", donor, description.size,
                     dense_seconds_since(started), changed, reachability.reachable, reachability.audited);
    }
    return dense_save(world, engine, engine.indexes[0], std::string(settings.save)) ? EXIT_SUCCESS : EXIT_FAILURE;
}

int dense_refine(world_t& world, int argc, char const* const* argv) {
    dense_settings_t settings;
    settings.max_refines = {.value = 1};
    std::array<bench::flag_t, 4> const flags {{
        {"--save", "address", "required: where the refined index is saved", &settings.save},
        {"--max-refines", "count", "defaults to 1: refinement passes, stopping early once settled",
         &settings.max_refines},
        {"--expansion-refine", "count", "defaults to the library's derivation: candidate pool while refining",
         &settings.expansion_refine},
        {"--threads", "count", "defaults to 7/8 of the domain's cores: executor threads", &settings.threads},
    }};
    std::array<std::string_view, 1> positionals;
    std::size_t given = 0;
    if (std::optional<int> const exit = parse_verb(
            world, argc, argv, "usearch dense refine <address> --save <address> [flags]", flags, positionals, given))
        return *exit;
    if (!runs_alone(world)) return EXIT_FAILURE;
    if (given != 1 || settings.save.empty()) {
        fmt::println(stderr, "a checkpoint and --save are required");
        return EXIT_FAILURE;
    }
    if (!checkpoints_admit_save(settings.save)) return EXIT_FAILURE;

    dense_engine_t engine;
    if (!dense_engine_for_ram(world, settings.threads, engine)) return EXIT_FAILURE;
    engine.indexes.resize(1);
    if (!dense_load(world, engine, positionals[0], engine.indexes[0])) return EXIT_FAILURE;
    if (!dense_refine_passes(world, engine, engine.indexes[0], settings.max_refines.value,
                             settings.expansion_refine.value))
        return EXIT_FAILURE;
    return dense_save(world, engine, engine.indexes[0], std::string(settings.save)) ? EXIT_SUCCESS : EXIT_FAILURE;
}

int dense_split(world_t& world, int argc, char const* const* argv) {
    dense_settings_t settings;
    std::array<bench::flag_t, 4> const flags {{
        {"--parts", "count", "required: receivers to split into, at least 2", &settings.parts},
        {"--save", "path", "required: the .safegraphs.json manifest to commit, its parts written beside it",
         &settings.save},
        {"--expansion-refine", "count", "defaults to the library's derivation: candidate pool while refining",
         &settings.expansion_refine},
        {"--threads", "count", "defaults to 7/8 of the domain's cores: executor threads", &settings.threads},
    }};
    std::array<std::string_view, 1> positionals;
    std::size_t given = 0;
    if (std::optional<int> const exit = parse_verb(
            world, argc, argv, "usearch dense split <address> --parts <count> --save <path>.safegraphs.json [flags]",
            flags, positionals, given))
        return *exit;
    if (!runs_alone(world)) return EXIT_FAILURE;
    if (given != 1 || !settings.parts.value || !settings.save.ends_with(dense_manifest_suffix_k)) {
        fmt::println(stderr, "a checkpoint, --parts and --save ending in {} are required", dense_manifest_suffix_k);
        return EXIT_FAILURE;
    }

    dense_engine_t engine;
    if (!dense_engine_for_ram(world, settings.threads, engine)) return EXIT_FAILURE;
    std::size_t const parts_count = settings.parts.value;
    engine.indexes.resize(1 + parts_count);
    usearch_dense_index_description_t description {};
    if (!dense_load(world, engine, positionals[0], engine.indexes[0])) return EXIT_FAILURE;
    if (!dense_describe(world, engine, engine.indexes[0], description)) return EXIT_FAILURE;

    usearch_outcome_t outcome {};
    usearch_dense_metadata_t const& metadata = description.metadata;
    for (std::size_t part = 0; part != parts_count; ++part) {
        usearch_dense_init_ram_t init {
            .index = &engine.indexes[1 + part],
            .outcome = &outcome,
            .metric = metadata.metric,
            .quantization = metadata.quantization,
            .dimensions = metadata.dimensions,
            .connectivity = metadata.connectivity,
            .ram = {&engine.placement, 1},
        };
        usearch_dense_init_ram(&init);
        if (dense_failed(world, outcome, "creating a receiver")) return EXIT_FAILURE;
    }

    dense_clock_t::time_point const started = dense_clock_t::now();
    std::size_t changed = 0;
    usearch_reachability_t reachability {};
    usearch_dense_split_t split {
        .index = engine.indexes[0],
        .executor = engine.executor,
        .outcome = &outcome,
        .count = parts_count,
        .receivers = engine.indexes.data() + 1,
        .changed = &changed,
        .reachability = &reachability,
        .expansion_factor = settings.expansion_refine.value,
    };
    usearch_dense_split(&split);
    if (dense_failed(world, outcome, "splitting")) return EXIT_FAILURE;
    fmt::println("split into {} parts in {:.1f} s, {} changed, {} of {} reachable", parts_count,
                 dense_seconds_since(started), changed, reachability.reachable, reachability.audited);

    std::string_view const stem = settings.save.substr(0, settings.save.size() - dense_manifest_suffix_k.size());
    std::vector<dense_part_t> parts(parts_count);
    for (std::size_t part = 0; part != parts_count; ++part) {
        std::string const part_path = dense_part_path(stem, part, parts_count);
        if (!dense_describe(world, engine, engine.indexes[1 + part], description)) return EXIT_FAILURE;
        if (!dense_save(world, engine, engine.indexes[1 + part], part_path)) return EXIT_FAILURE;
        parts[part] = {dense_file_bytes(part_path), description.size};
    }
    if (!dense_write_manifest(settings.save, parts)) {
        fmt::println(stderr, "cannot write {}", settings.save);
        return EXIT_FAILURE;
    }
    fmt::println("committed {} over {} parts", settings.save, parts_count);
    return EXIT_SUCCESS;
}

int dense_health(world_t& world, int argc, char const* const* argv) {
    dense_settings_t settings;
    std::array<bench::flag_t, 1> const flags {{
        {"--threads", "count", "defaults to the devices' queue capabilities: executor threads", &settings.threads},
    }};
    std::array<std::string_view, 1> positionals;
    std::size_t given = 0;
    if (std::optional<int> const exit = parse_verb(world, argc, argv, "usearch dense health <address> [flags]", flags,
                                                   positionals, given))
        return *exit;
    if (!runs_alone(world)) return EXIT_FAILURE;
    if (given != 1) {
        fmt::println(stderr, "a checkpoint is required");
        return EXIT_FAILURE;
    }

    // A view reads the graph from storage, so the audit holds nothing that grows with the index.
    dense_engine_t engine;
    usearch_outcome_t outcome {};
    usearch_executor_for_nvme_t executor {.executor = &engine.executor,
                                          .outcome = &outcome,
                                          .threads = settings.threads.value};
    usearch_executor_for_nvme(&executor);
    if (dense_failed(world, outcome, "creating the NVMe executor")) return EXIT_FAILURE;
    engine.indexes.resize(1);
    usearch_dense_view_nvme_t view {.index = &engine.indexes[0],
                                    .executor = engine.executor,
                                    .outcome = &outcome,
                                    .path = dense_bytes(positionals[0])};
    usearch_dense_view_nvme(&view);
    if (dense_failed(world, outcome, fmt::format("viewing {}", positionals[0]))) return EXIT_FAILURE;

    dense_clock_t::time_point const started = dense_clock_t::now();
    usearch_reachability_t reachability {};
    usearch_dense_health_t health {.index = engine.indexes[0],
                                   .executor = engine.executor,
                                   .outcome = &outcome,
                                   .reachability = &reachability};
    usearch_dense_health(&health);
    if (dense_failed(world, outcome, "auditing")) return EXIT_FAILURE;
    fmt::println("{}: {} of {} reachable over {} sweeps in {:.1f} s", positionals[0], reachability.reachable,
                 reachability.audited, reachability.sweeps, dense_seconds_since(started));
    return EXIT_SUCCESS;
}

} // namespace unum::usearch::tools
