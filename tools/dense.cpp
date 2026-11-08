/**
 *  @file tools/dense.cpp
 *  @author Ash Vardanian
 *  @date October 17, 2026
 *  @brief The dense verbs: insert, absorb, refine, split, health, search and get, each over
 *      SafeGraphs checkpoints at any address.
 *
 *  @c insert indexes a contiguous slice of one RetriEval file or Arrow input on a GPU or in RAM and
 *  saves it, @c absorb merges checkpoints into one, donor by donor, @c refine re-selects a
 *  checkpoint's neighbors, @c split cuts one into the parts of a manifest, @c health audits how
 *  much of one a search reaches, @c search queries one with the rows of another RetriEval or Arrow
 *  input, and @c get retrieves the vectors of an Arrow column of keys. Keys come row-aligned from
 *  @c --keys or @c --keys-column, or are the dataset row numbers, so slices built apart keep
 *  distinct keys when merged.
 *
 *  An Arrow input reads a path or standard input, sniffed by its first bytes, and its columns reach
 *  the library through the converters of @c usearch/arrow.h. A checkpoint or an Arrow output may
 *  take @c - for a pipe, whose verb then prints its steps to standard error.
 *
 *  Across several ranks, @c insert is a distributed build: rank r of R takes rows ⌊N·r/R⌋ to
 *  ⌊N·(r+1)/R⌋ of the slice, saves them as one part under the name a sharded save gives it, and
 *  once every rank reports its part complete, rank 0 writes the manifest beside them, the commit
 *  record, so a run failing anywhere leaves parts and no checkpoint. Ranks agree on their slices
 *  from the world size alone, and talk only to stop together and to gather the parts.
 */
#include <cstdint> // `std::uint64_t`, `std::int32_t`
#include <cstdio>  // `stderr`
#include <cstdlib> // `EXIT_SUCCESS`, `EXIT_FAILURE`

#include <fcntl.h>    // `open`, `O_RDONLY`
#include <sys/stat.h> // `fstat`
#include <unistd.h>   // `pread`, `close`

#include <algorithm>   // `std::min`
#include <array>       // `std::array`
#include <limits>      // `std::numeric_limits`
#include <string>      // `std::string`
#include <string_view> // `std::string_view`
#include <vector>      // `std::vector`

#include <fmt/format.h>

#include <forkunion/forkunion.h> // `fu_topology_new`, `fu_memory_domain_id_at_index`

#include "usearch/arrow.h"
#include "usearch/dense_index.h"

#include "abi.hpp"   // `count_t`, `devices_from_environment`, `session_t`, `handle`
#include "world.hpp" // `world_t`, `parse_verb`, `checkpoint_save`, `checkpoint_split`

namespace unum::usearch::tools {

/** Two little-endian `u32` counters, rows then columns, open every RetriEval file. */
static constexpr std::size_t dense_header_bytes_k = 8;

/** Every setting the insert takes beyond the checkpoint verbs', each spelled as its flag with
 *  dashes. */
struct dense_settings_t {
    checkpoint_settings_t checkpoint;
    std::string_view keys;
    std::string_view stored_dtype;
    std::string_view metric = "angular";
    std::string_view input;
    std::string_view vectors_column;
    std::string_view keys_column;
};

/** What @ref dense_source_t::rows holds for an input read to its end, sized by nothing up front. */
static constexpr std::size_t dense_rows_unknown_k = std::numeric_limits<std::size_t>::max();

/** The Arrow input an insert reads, its columns, and the first non-empty batch already read. */
struct dense_arrow_t {

    /** The input, on the batch the insert takes next. */
    arrow_input_t input;

    /** Where the rows sit among its columns. */
    std::size_t vectors_column = arrow_no_column_k;

    /** Where the keys sit among its columns, none for row numbers. */
    std::size_t keys_column = arrow_no_column_k;

    /** Row-number or widened keys of the rows the insert takes next. */
    std::vector<usearch_key_t> keys;
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

/** Prints a failed call's status and message under the rank that saw it; true when it failed. */
static bool dense_failed(world_t const& world, usearch_outcome_t const& outcome, std::string_view verb) {
    return report_failure(outcome, fmt::format("{}{}", rank_label(world), verb));
}

/** Opens @p path and checks its counters against its size. */
static bool dense_open_vectors(world_t const& world, std::string_view path, dense_source_t& source) {
    source.scalar = scalar_of_extension(path);
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
    source.row_bytes = row_bytes_of(source.scalar, source.columns);
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
 *  integer per row, as every RetriEval scalar is stored, filling a key's low bytes as row numbers
 *  do; a 128-bit integer fills the whole key, so a UUID's integer form keys its row. */
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
    std::vector<std::int32_t> const listed = devices_from_environment();
    if (listed.empty()) {
        device = static_cast<std::int32_t>(world.local_rank % devices_count);
        return true;
    }
    device = listed[world.local_rank % listed.size()];
    if (static_cast<std::size_t>(device) < devices_count) return true;
    fmt::println(stderr, "{}USEARCH_DEVICES names device {}, past the {} listed", rank_label(world), device,
                 devices_count);
    return false;
}

/** Creates a CPU executor over the memory domain at the node-local rank's position, wrapping
 *  around, so ranks sharing a node spread over its sockets; the index lives there as one shard,
 *  so a saved monolith stays one graph. */
static bool dense_session_for_ram(world_t const& world, count_t const& threads, session_t& session) {
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
    session.residency = usearch_residency_ram_k;
    session.memory_domains = {static_cast<std::int32_t>(found)};
    std::vector<std::int32_t> domains;
    usearch_outcome_t outcome {};
    usearch_executor_for_ram_t executor {.executor = &session.executor,
                                         .outcome = &outcome,
                                         .ram = session.ram_options(1, domains),
                                         .threads = threads.value};
    usearch_executor_for_ram(&executor);
    return !dense_failed(world, outcome, "creating the CPU executor");
}

/** Inserts the slice, reading one staging while the index ingests the other. */
static bool dense_insert_slice(world_t const& world, dense_settings_t const& settings, dense_source_t const& source,
                               handle<dense_verbs_t> const& index) {
    std::size_t const batch_rows = std::min(settings.checkpoint.batch.value, source.slice_rows);
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
        else
            for (std::size_t position = 0; position != count; ++position)
                staging.keys[position] = std::bit_cast<usearch_key_t>(uuid_of(row + position));
        usearch_dense_insert_t insert {
            .index = index.get(),
            .executor = index.executor(),
            .outcome = &staging.outcome,
            .completion = usearch_completion_deferred_k,
            .count = count,
            .keys = {staging.keys.data(), sizeof(usearch_key_t)},
            .vectors = {staging.rows.data(), source.row_bytes, source.columns, source.scalar},
            .expansion_factor = settings.checkpoint.expansion_add.value,
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

/** Inserts the rows of @p input's batch from @p first_position on, in calls of at most
 *  <tt>--batch</tt> rows, the keys of its keys column or the row numbers from @p first_row on. */
static bool dense_insert_batch(world_t const& world, dense_settings_t const& settings, dense_arrow_t& arrow,
                               handle<dense_verbs_t> const& index, std::size_t first_position, std::size_t count,
                               std::uint64_t first_row) {
    usearch_rows_view_t rows {};
    usearch_keys_view_t keys {};
    std::string failure = arrow_rows_of(arrow.input, arrow.vectors_column, rows);
    if (failure.empty() && arrow.keys_column != arrow_no_column_k)
        failure = arrow_keys_of(arrow.input, arrow.keys_column, arrow.keys, keys);
    else if (failure.empty()) {
        arrow.keys.resize(first_position + count);
        for (std::size_t position = 0; position != count; ++position)
            arrow.keys[first_position + position] = std::bit_cast<usearch_key_t>(uuid_of(first_row + position));
        keys = {arrow.keys.data(), sizeof(usearch_key_t)};
    }
    if (!failure.empty()) {
        fmt::println(stderr, "{}{}", rank_label(world), failure);
        return false;
    }
    std::size_t const batch_rows = std::min(settings.checkpoint.batch.value, count);
    for (std::size_t offset = 0; offset < count; offset += batch_rows) {
        std::size_t const position = first_position + offset;
        usearch_outcome_t outcome {};
        usearch_dense_insert_t insert {
            .index = index.get(),
            .executor = index.executor(),
            .outcome = &outcome,
            .count = std::min(batch_rows, count - offset),
            .keys = {keys.data + position, keys.stride},
            .vectors = {static_cast<usearch_byte_t const*>(rows.data) + position * rows.stride, rows.stride,
                        rows.columns, rows.scalar},
            .expansion_factor = settings.checkpoint.expansion_add.value,
        };
        usearch_dense_insert(&insert);
        if (dense_failed(world, outcome, "insert")) return false;
    }
    return true;
}

/** Inserts the slice of an Arrow input batch by batch, the columns read in place, and answers how
 *  many rows it took into @p source; an input shorter than an explicit slice fails. */
static bool dense_insert_arrow(world_t const& world, dense_settings_t const& settings, dense_source_t& source,
                               dense_arrow_t& arrow, handle<dense_verbs_t> const& index) {
    std::uint64_t const begin = source.first_row;
    std::uint64_t const end = source.slice_rows == dense_rows_unknown_k ? dense_rows_unknown_k
                                                                        : begin + source.slice_rows;
    std::uint64_t row = 0;
    for (bool more = true; more && row < end;) {
        std::uint64_t const batch_end = row + arrow.input.rows();
        std::uint64_t const taken_begin = std::max(row, begin), taken_end = std::min(batch_end, end);
        if (taken_begin < taken_end &&
            !dense_insert_batch(world, settings, arrow, index, static_cast<std::size_t>(taken_begin - row),
                                static_cast<std::size_t>(taken_end - taken_begin), taken_begin))
            return false;
        row = batch_end;
        if (row >= end) break;
        if (std::string const failure = arrow.input.next(more); !failure.empty()) {
            fmt::println(stderr, "{}{}", rank_label(world), failure);
            return false;
        }
    }
    if (row <= begin || (end != dense_rows_unknown_k && row < end)) {
        fmt::println(stderr, "{}--first-entry={} --entries={} reach past the {} rows of {}", rank_label(world),
                     settings.checkpoint.first_entry.value, settings.checkpoint.entries.value, row, arrow.input.name());
        return false;
    }
    source.slice_rows = static_cast<std::size_t>(std::min(row, end) - begin);
    return true;
}

/** Opens the Arrow input, picks its columns and reads its first non-empty batch, which sets the
 *  rows' scalar and width; across several ranks, walks a second reading to count the rows they
 *  share, and otherwise leaves the count unknown, as a pipe has it. */
static bool dense_open_arrow(world_t const& world, dense_settings_t const& settings, dense_source_t& source,
                             dense_arrow_t& arrow) {
    std::string failure = arrow.input.open(settings.input);
    if (failure.empty()) failure = arrow_pick_rows(arrow.input, settings.vectors_column, arrow.vectors_column);
    if (failure.empty() && !settings.keys_column.empty())
        failure = arrow_pick_column(arrow.input, settings.keys_column, arrow.keys_column);
    bool more = failure.empty();
    while (failure.empty() && more && (failure = arrow.input.next(more)).empty() && more && !arrow.input.rows()) {}
    if (failure.empty() && !more) failure = fmt::format("{} holds no rows", arrow.input.name());
    usearch_rows_view_t rows {};
    if (failure.empty()) failure = arrow_rows_of(arrow.input, arrow.vectors_column, rows);
    if (!failure.empty()) {
        fmt::println(stderr, "{}{}", rank_label(world), failure);
        return false;
    }
    source.scalar = rows.scalar, source.columns = rows.columns, source.row_bytes = rows.stride;
    source.rows = dense_rows_unknown_k;
    if (world.size == 1) return true;

    arrow_input_t counted;
    source.rows = 0;
    failure = counted.open(settings.input);
    for (more = failure.empty(); failure.empty() && more;) {
        failure = counted.next(more);
        source.rows += more ? counted.rows() : 0;
    }
    if (!failure.empty()) fmt::println(stderr, "{}{}", rank_label(world), failure);
    return failure.empty();
}

/** Opens the vectors, takes this rank's share of the slice, creates the executor and the empty
 *  one-shard index on the residency asked for, and reserves room for the share's entries where its
 *  count is known, stopping at the first failure. */
static bool dense_insert_setup(world_t const& world, dense_settings_t const& settings, std::string_view vectors,
                               usearch_dense_metric_t metric, dense_source_t& source, dense_arrow_t& arrow,
                               session_t& session, handle<dense_verbs_t>& index) {
    std::string const rank = rank_label(world);
    if (settings.input.empty()) {
        if (!dense_open_vectors(world, vectors, source)) return false;
        if (!settings.keys.empty() && !dense_open_keys(world, settings.keys, source)) return false;
    }
    else if (!dense_open_arrow(world, settings, source, arrow)) return false;
    std::uint64_t const first = settings.checkpoint.first_entry.value;
    std::uint64_t const entries = settings.checkpoint.entries.value;
    if (first >= source.rows || entries > source.rows - first) {
        fmt::println(stderr, "{}--first-entry={} --entries={} reach past the {} rows of {}", rank, first, entries,
                     source.rows, vectors);
        return false;
    }
    // Ranks share the slice as evenly as integers allow, so every share is known without asking.
    std::uint64_t const slice = entries ? entries : source.rows - first;
    source.first_row = first + slice * world.rank / world.size;
    source.slice_rows = static_cast<std::size_t>(first + slice * (world.rank + 1) / world.size - source.first_row);
    bool const unbounded = source.rows == dense_rows_unknown_k && !entries;
    if (unbounded) source.slice_rows = dense_rows_unknown_k;
    if (!source.slice_rows) {
        fmt::println(stderr, "{}{} rows leave no share for each of {} ranks", rank, slice, world.size);
        return false;
    }

    usearch_scalar_t quantization = source.scalar;
    if (!settings.stored_dtype.empty()) {
        quantization = usearch_scalar_from_name(bytes_view(settings.stored_dtype));
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

    usearch_residency_t const asked = settings.checkpoint.session.residency;
    session.residency = asked != usearch_residency_unknown_k
                            ? asked
                            : (devices_count ? usearch_residency_vram_k : usearch_residency_ram_k);
    if (session.residency == usearch_residency_nvme_k ||
        (session.residency == usearch_residency_vram_k && !devices_count)) {
        fmt::println(stderr, "{}--residency=\"{}\" is neither ram nor vram over a listed GPU", rank,
                     usearch_residency_name(session.residency));
        return false;
    }

    if (session.residency == usearch_residency_vram_k) {
        std::int32_t device = 0;
        if (!dense_device_of(world, devices_count, device)) return false;
        session.devices = {device};
        std::vector<std::int32_t> ordinals;
        usearch_executor_for_vram_t executor {.executor = &session.executor,
                                              .outcome = &outcome,
                                              .vram = session.vram_options(1, ordinals)};
        usearch_executor_for_vram(&executor);
        if (dense_failed(world, outcome, "creating the GPU executor")) return false;
    }
    else if (!dense_session_for_ram(world, settings.checkpoint.session.threads, session)) return false;
    usearch_arena_init_t arena {
        .arena = &session.arena,
        .executor = session.residency == usearch_residency_vram_k ? session.executor : nullptr,
        .outcome = &outcome,
    };
    usearch_arena_init(&arena);
    if (dense_failed(world, outcome, "opening the arena")) return false;
    index = open_index<dense_verbs_t>(
        session, {metric, quantization, source.columns, settings.checkpoint.connectivity.value}, 1);
    if (!index) return false;

    if (unbounded) return true;
    usearch_dense_reserve_t reserve {.index = index.get(),
                                     .executor = index.executor(),
                                     .outcome = &outcome,
                                     .capacity = source.slice_rows};
    usearch_dense_reserve(&reserve);
    return !dense_failed(world, outcome, "reserving");
}

/** Inserts this rank's share, refines it and saves it to @p address, printing each step to
 *  @p steps. */
static bool dense_construct(world_t const& world, dense_settings_t const& settings, dense_source_t& source,
                            dense_arrow_t& arrow, handle<dense_verbs_t> const& index, std::string const& address,
                            checkpoint_clock_t::time_point started, std::FILE* steps) {
    bool const inserted = settings.input.empty() ? dense_insert_slice(world, settings, source, index)
                                                 : dense_insert_arrow(world, settings, source, arrow, index);
    if (!inserted) return false;
    bool const on_device = index.residency == usearch_residency_vram_k;
    fmt::println(steps, "{}inserted rows {} to {} on {} {} {} in {:.1f} s", rank_label(world), source.first_row,
                 source.first_row + source.slice_rows, usearch_residency_name(index.residency),
                 on_device ? "device" : "memory domain",
                 on_device ? index.session->devices[0] : index.session->memory_domains[0],
                 checkpoint_seconds_since(started));
    checkpoint_settings_t const& checkpoint = settings.checkpoint;
    return checkpoint_refine_passes(world, index, checkpoint.max_refines.value,
                                    {checkpoint.expansion_refine.value, checkpoint.memory_limit.value}, steps) &&
           checkpoint_save(world, index, address, steps);
}

int modality_insert(dense_verbs_t, world_t& world, int argc, char const* const* argv) {
    dense_settings_t settings;
    settings.checkpoint.session.residency = usearch_residency_unknown_k;
    std::array<flag_t, 6> const modality_flags {{
        {"--keys", "path", "defaults to row numbers: a .u64bin or .u128bin of one key per vector row", &settings.keys},
        {"--input", "path", "an Arrow file or stream in place of a RetriEval file, or - for standard input",
         &settings.input},
        {"--vectors-column", "name", "defaults to the only fixed-size column of rows in --input",
         &settings.vectors_column},
        {"--keys-column", "name", "defaults to row numbers: a uuid, uint64 or int64 column of --input",
         &settings.keys_column},
        {"--stored-dtype", "name",
         "defaults to the file's: bf16, f16, e5m2, e4m3, e3m2, e2m3, i8, i4x2, u8, u4x2 or u1x8",
         &settings.stored_dtype},
        {"--metric", "name", "defaults to angular: inner, angular, euclidean, hamming or jaccard", &settings.metric},
    }};
    std::vector<flag_t> const flags = insert_flags(settings.checkpoint, modality_flags);
    std::array<std::string_view, 1> positionals;
    std::size_t given = 0;
    if (std::optional<int> const exit = parse_verb(
            world, argc, argv, "usearch dense insert <vectors|--input path> --save <address|-> [flags]", flags,
            positionals, given))
        return *exit;
    bool const distributed = world.size > 1;
    bool const speaks = world.rank == 0;
    std::string_view const save = settings.checkpoint.save;
    bool const from_arrow = !settings.input.empty();
    if (given + from_arrow != 1 || save.empty()) {
        if (speaks) fmt::println(stderr, "the vectors or --input, and --save, are required");
        return EXIT_FAILURE;
    }
    if (from_arrow ? !settings.keys.empty() : !settings.keys_column.empty() || !settings.vectors_column.empty()) {
        if (speaks) fmt::println(stderr, "--keys reads a RetriEval file's keys, and the column flags --input's");
        return EXIT_FAILURE;
    }
    if (distributed && (settings.input == "-" || save == "-")) {
        if (speaks) fmt::println(stderr, "- is one pipe, which several ranks cannot share");
        return EXIT_FAILURE;
    }
    usearch_dense_metric_t const metric = usearch_dense_metric_from_name(bytes_view(settings.metric));
    if (metric == usearch_dense_metric_unknown_k) {
        if (speaks) fmt::println(stderr, "--metric=\"{}\" names no metric", settings.metric);
        return EXIT_FAILURE;
    }
    std::optional<std::string> const admitted = distributed ? std::optional<std::string> {save}
                                                            : checkpoints_admit_save(save);
    if (!admitted) return EXIT_FAILURE;

    std::FILE* const steps = save == "-" ? stderr : stdout;
    checkpoint_clock_t::time_point const started = checkpoint_clock_t::now();
    dense_source_t source;
    dense_arrow_t arrow;
    session_t session;
    handle<dense_verbs_t> index;
    // One rank unable to start stops every rank before any of them spends hours building.
    if (world.barrier(!dense_insert_setup(world, settings, positionals[0], metric, source, arrow, session, index)))
        return EXIT_FAILURE;

    std::string const address = distributed ? checkpoint_part_path(save, world.rank, world.size) : *admitted;
    bool const built = dense_construct(world, settings, source, arrow, index, address, started, steps);
    if (world.barrier(!built)) return EXIT_FAILURE;
    if (!distributed) return EXIT_SUCCESS;

    std::array<std::uint64_t, 2> const sent {checkpoint_file_bytes(address), source.slice_rows};
    std::vector<std::uint64_t> gathered(speaks ? sent.size() * world.size : 0);
    world.gather(sent, gathered);
    if (!speaks) return EXIT_SUCCESS;
    std::vector<safegraphs_part_t> parts(world.size);
    for (std::size_t part = 0; part != parts.size(); ++part)
        parts[part] = {.bytes = gathered[part * sent.size()], .nodes = gathered[part * sent.size() + 1]};
    if (!checkpoint_write_manifest(save, parts)) {
        fmt::println(stderr, "rank 0: cannot write {}", save);
        return EXIT_FAILURE;
    }
    std::size_t const entries = settings.checkpoint.entries.value;
    fmt::println("committed {} over {} parts of {} entries in {:.1f} s", save, world.size,
                 entries ? entries : source.rows - settings.checkpoint.first_entry.value,
                 checkpoint_seconds_since(started));
    return EXIT_SUCCESS;
}

int modality_split(dense_verbs_t, world_t& world, int argc, char const* const* argv) {
    return checkpoint_split<dense_verbs_t>(world, argc, argv, {},
                                           [](usearch_dense_metadata_t const& metadata) { return metadata; });
}

/** Gathers the rows of the Arrow @p input into @p rows, and its keys when @p keys_column names a
 *  column, as one search's queries; prints why not. */
static std::optional<checkpoint_queries<dense_verbs_t>> dense_read_queries(std::string_view input,
                                                                           std::string_view vectors_column,
                                                                           std::string_view keys_column,
                                                                           std::vector<usearch_byte_t>& rows) {
    arrow_input_t source;
    std::size_t vectors = arrow_no_column_k, keys = arrow_no_column_k;
    std::string failure = source.open(input);
    if (failure.empty()) failure = arrow_pick_rows(source, vectors_column, vectors);
    if (failure.empty() && !keys_column.empty()) failure = arrow_pick_column(source, keys_column, keys);

    checkpoint_queries<dense_verbs_t> queries;
    std::vector<usearch_key_t> widened;
    while (failure.empty()) {
        bool more = false;
        failure = source.next(more);
        if (!failure.empty() || !more) break;
        usearch_rows_view_t batch {};
        usearch_keys_view_t batch_keys {};
        failure = arrow_rows_of(source, vectors, batch);
        if (failure.empty() && keys != arrow_no_column_k) failure = arrow_keys_of(source, keys, widened, batch_keys);
        if (!failure.empty()) break;
        if (queries.count && (batch.columns != queries.queries.columns || batch.scalar != queries.queries.scalar)) {
            failure = fmt::format("{} changes its rows' type between batches", source.name());
            break;
        }
        queries.queries.columns = batch.columns, queries.queries.scalar = batch.scalar;
        queries.queries.stride = batch.stride;
        auto const* const bytes = static_cast<usearch_byte_t const*>(batch.data);
        for (std::size_t row = 0; row != source.rows(); ++row) {
            rows.insert(rows.end(), bytes + row * batch.stride, bytes + (row + 1) * batch.stride);
            if (keys != arrow_no_column_k) queries.keys.push_back(batch_keys.data[row]);
        }
        queries.count += source.rows();
    }
    if (!failure.empty()) {
        fmt::println(stderr, "{}", failure);
        return std::nullopt;
    }
    queries.queries.data = rows.data();
    return queries;
}

int modality_search(dense_verbs_t, world_t& world, int argc, char const* const* argv) {
    std::string_view query_vectors, input, vectors_column, keys_column, index_address;
    std::array<flag_t, 5> const flags {{
        {"--query-vectors", "path", "a RetriEval file of one query per row, unless --input is given", &query_vectors},
        {"--input", "path", "an Arrow file or stream of queries, or - for standard input", &input},
        {"--vectors-column", "name", "defaults to the only fixed-size column of rows in --input", &vectors_column},
        {"--keys-column", "name", "defaults to row numbers: a uuid, uint64 or int64 column of --input", &keys_column},
        {"--index", "address", "the checkpoint in place of the positional, or - for standard input", &index_address},
    }};
    mapped_rows_t queries_file;
    std::vector<usearch_byte_t> query_rows;
    return checkpoint_search<dense_verbs_t>(
        world, argc, argv, "usearch dense search <address> --query-vectors <path> [flags]", flags,
        [&]() -> std::optional<checkpoint_queries<dense_verbs_t>> {
            if (query_vectors.empty() == input.empty()) {
                fmt::println(stderr, "exactly one of --query-vectors and --input is required");
                return std::nullopt;
            }
            if (input == "-" && index_address == "-") {
                fmt::println(stderr, "--input and --index cannot both read standard input");
                return std::nullopt;
            }
            if (!input.empty()) return dense_read_queries(input, vectors_column, keys_column, query_rows);
            auto [file, status] = mapped_rows_t::open(query_vectors);
            queries_file = std::move(file);
            usearch_rows_view_t const rows = queries_file.view(0, queries_file.rows());
            if (failed(status) || queries_file.scalar() == usearch_scalar_unknown_k || !rows.data) {
                fmt::println(stderr,
                             "--query-vectors=\"{}\" is no .fbin, .bf16bin, .f16bin, .i8bin, .u8bin or .u1x8bin",
                             query_vectors);
                return std::nullopt;
            }
            return checkpoint_queries<dense_verbs_t> {rows, queries_file.rows()};
        },
        index_address);
}

/** The rows a get gathered: every key once per entry, the entries' bytes packed, and the schema of
 *  their column, taken from the library's first export. */
struct dense_retrieved_t {
    std::vector<usearch_key_t> keys;
    std::vector<usearch_byte_t> rows;
    ArrowSchema vector {};

    dense_retrieved_t() = default;
    dense_retrieved_t(dense_retrieved_t const&) = delete;
    dense_retrieved_t& operator=(dense_retrieved_t const&) = delete;
    ~dense_retrieved_t() noexcept {
        if (vector.release) vector.release(&vector);
    }
};

/** Retrieves every vector under @p key out of @p index into @p retrieved; false after printing
 *  why. */
static bool dense_get_key(handle<dense_verbs_t> const& index, usearch_key_t key, usearch_scalar_t outputs,
                          dense_retrieved_t& retrieved) {
    usearch_outcome_t outcome {};
    usearch_rows_view_t vectors {};
    std::size_t written = 0;
    usearch_dense_get_t get {
        .index = index.get(),
        .executor = index.executor(),
        .outcome = &outcome,
        .key = key,
        .vectors = &vectors,
        .written = &written,
        .arena = index.session->arena,
        .outputs_scalar = outputs,
    };
    usearch_dense_get(&get);
    if (report_failure(outcome, "getting")) return false;
    if (!written) return true;
    if (!retrieved.vector.release) {
        ArrowDeviceArray exported {};
        usearch_status_t const status = usearch_rows_view_to_arrow(vectors, written, index.session->arena,
                                                                   &retrieved.vector, &exported);
        if (status != usearch_status_success_k) {
            fmt::println(stderr, "exporting rows failed with {}", usearch_status_name(status));
            return false;
        }
        // The next get rewinds the arena, which the export pins until its array lets go.
        exported.array.release(&exported.array);
    }
    std::size_t const offset = retrieved.rows.size(), bytes = written * vectors.stride;
    retrieved.rows.resize(offset + bytes);
    if (!download(retrieved.rows.data() + offset, vectors.data, bytes)) {
        fmt::println(stderr, "cannot read the rows back off their device");
        return false;
    }
    retrieved.keys.insert(retrieved.keys.end(), written, key);
    return true;
}

int dense_get(world_t& world, int argc, char const* const* argv) {
    checkpoint_settings_t settings;
    std::string_view index_address, input, keys_column, outputs_dtype;
    std::vector<flag_t> const flags = with_session_flags(
        settings.session,
        {
            {"--index", "address", "required: the checkpoint at any address, or - for standard input", &index_address},
            {"--input", "path", "required: an Arrow file or stream of keys, or - for standard input", &input},
            {"--keys-column", "name", "required: a uuid, uint64 or int64 column of --input", &keys_column},
            {"--outputs-dtype", "name", "defaults to the checkpoint's: the scalar to convert the vectors to",
             &outputs_dtype},
            {"--output", "path",
             "required: where the keys and vectors land as Arrow, a stream for - or an .arrows path, else a file",
             &settings.output},
        });
    std::array<std::string_view, 1> positionals;
    std::size_t given = 0;
    if (std::optional<int> const exit = parse_verb(
            world, argc, argv,
            "usearch dense get --index <address|-> --input <path|-> --keys-column <name> --output <path|-> [flags]",
            flags, positionals, given))
        return *exit;
    if (!runs_alone(world)) return EXIT_FAILURE;
    if (given || index_address.empty() || input.empty() || keys_column.empty() || settings.output.empty()) {
        fmt::println(stderr, "--index, --input, --keys-column and --output are required, and nothing else");
        return EXIT_FAILURE;
    }
    if (input == "-" && index_address == "-") {
        fmt::println(stderr, "--input and --index cannot both read standard input");
        return EXIT_FAILURE;
    }
    usearch_scalar_t outputs = usearch_scalar_unknown_k;
    if (!outputs_dtype.empty()) {
        outputs = usearch_scalar_from_name(bytes_view(outputs_dtype));
        if (outputs == usearch_scalar_unknown_k) {
            fmt::println(stderr, "--outputs-dtype=\"{}\" names no scalar", outputs_dtype);
            return EXIT_FAILURE;
        }
    }
    std::FILE* const steps = settings.output == "-" ? stderr : stdout;

    arrow_input_t source;
    std::size_t keys = arrow_no_column_k;
    std::string failure = source.open(input);
    if (failure.empty()) failure = arrow_pick_column(source, keys_column, keys);
    if (!failure.empty()) {
        fmt::println(stderr, "{}", failure);
        return EXIT_FAILURE;
    }
    settings.session.devices = devices_from_environment();
    session_t session = open_session(settings.session, usearch_cpu_class_unknown_k);
    if (!session) return EXIT_FAILURE;
    handle<dense_verbs_t> index = checkpoint_open<dense_verbs_t>(session, index_address, settings.session.residency, {},
                                                                 steps);
    if (!index) return EXIT_FAILURE;
    if (outputs == usearch_scalar_unknown_k) outputs = describe(index).metadata.quantization;

    checkpoint_clock_t::time_point const started = checkpoint_clock_t::now();
    dense_retrieved_t retrieved;
    std::vector<usearch_key_t> widened;
    std::size_t asked = 0, missing = 0;
    while (true) {
        bool more = false;
        failure = source.next(more);
        if (!failure.empty() || !more) break;
        usearch_keys_view_t batch_keys {};
        failure = arrow_keys_of(source, keys, widened, batch_keys);
        if (!failure.empty()) break;
        for (std::size_t row = 0; row != source.rows(); ++row) {
            std::size_t const held = retrieved.keys.size();
            if (!dense_get_key(index, batch_keys.data[row], outputs, retrieved)) return EXIT_FAILURE;
            missing += retrieved.keys.size() == held;
        }
        asked += source.rows();
    }
    if (failure.empty() && retrieved.keys.empty()) failure = "none of the keys is in the index";
    if (failure.empty())
        failure = arrow_vectors_write(settings.output, retrieved.keys, retrieved.vector,
                                      std::as_bytes(std::span {retrieved.rows}));
    if (!failure.empty()) {
        fmt::println(stderr, "{}", failure);
        return EXIT_FAILURE;
    }
    fmt::println(steps, "retrieved {} vectors of {} keys in {:.3f} s, {} keys missing", retrieved.keys.size(), asked,
                 checkpoint_seconds_since(started), missing);
    return EXIT_SUCCESS;
}

} // namespace unum::usearch::tools
