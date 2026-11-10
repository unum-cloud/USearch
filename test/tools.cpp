/**
 *  @file test/tools.cpp
 *  @author Ash Vardanian
 *  @date November 1, 2026
 *  @brief The command-line tools' suite: a checkpoint of two parts of every modality exports to
 *      Apache Arrow and imports back alike, and a file standing as a drive maps its slots.
 *
 *  The export writes one record batch per part, its keys as the UUID extension and dense vectors as
 *  fixed-size lists, and the checkpoint the import lays out loads with the original's description,
 *  answers the original's searches, and saves again byte for byte, shapes within their rounding.
 *  Those searches' matches read back from Arrow alike in either framing. The drive's slots hold a
 *  checkpoint, copied on as @c usearch @c copy does, then zeros, then bytes of no format.
 *
 *  Arrow reads and writes standard input and output through real pipes, sniffing either framing,
 *  and refuses a cut stream and Parquet. With @c USEARCH_BINARY naming the built @c usearch, an
 *  insert, a metadata read, a search and a get chain through those pipes as a shell would.
 */
#include <cstdint> // `std::uint32_t` - the seed in a file's name
#include <cstdio>  // `std::remove`, `popen` - the round trip's files and the pipes
#include <cstdlib> // `std::getenv`, `std::system` - where the round trip writes, and the tool it runs
#include <cstring> // `std::memset` - the foreign slot

#include <array>        // `std::array` - the footer's fields
#include <bit>          // `std::bit_cast` - query numbers into keys
#include <filesystem>   // `std::filesystem::resize_file` - the file standing as a drive
#include <limits>       // `std::numeric_limits` - the unrecorded fields of an observation
#include <string>       // `std::string` - the paths and the imported checkpoint
#include <string_view>  // `std::string_view` - the framings' extensions
#include <system_error> // `std::error_code` - the drive's size
#include <type_traits>  // `std::is_same_v` - what only dense metadata records
#include <vector>       // `std::vector` - the footer's blocks

#include "usearch/usearch.h"

#include "arrow_checkpoints.hpp" // `arrow_export`, `arrow_import`, `arrow_matches_write`
#include "harness.hpp"
#include "slots.hpp"

#if !USEARCH_OS_WINDOWS_
#include <unistd.h> // `dup`, `dup2` - the standard descriptors the pipes stand in for
#endif

namespace unum::usearch::test::abi {

/** Entries the index holds over its two shards, the dimensions of each dense one, and the queries
 *  both searches answer. */
static constexpr std::size_t arrow_entries_k = 1024, arrow_dimensions_k = 64, arrow_queries_k = 128;

/** Parts the saved checkpoint spans, one per shard. */
static constexpr std::size_t arrow_parts_k = 2;

/** Bytes of each sequence, and points of each cloud. */
static constexpr std::size_t arrow_sequence_bytes_k = 64, arrow_cloud_points_k = 8;

/** What a modality's round trip builds and searches, the entries answering as queries. */
template <typename verbs_type_>
struct arrow_fixture;

template <>
struct arrow_fixture<dense_verbs_t> {
    nk::matrix<nk::f32_t> vectors;
    usearch_dense_metadata_t geometry {usearch_dense_metric_angular_k, usearch_scalar_bf16_k, arrow_dimensions_k, 0};

    explicit arrow_fixture(u64_t seed) : vectors(make_vectors<nk::f32_t>(arrow_entries_k, arrow_dimensions_k, seed)) {}
    usearch_rows_view_t values() const noexcept {
        return {vectors.data(), static_cast<std::size_t>(vectors.stride_bytes(0)), arrow_dimensions_k,
                usearch_scalar_f32_k};
    }
    usearch_rows_view_t queries() const noexcept { return values(); }
};

template <>
struct arrow_fixture<sequence_verbs_t> {
    std::string symbols;
    std::vector<std::size_t> offsets {0};
    sequence_geometry_t geometry {{usearch_sequence_metric_levenshtein_bytes_k, 0}, nullptr, {}};

    explicit arrow_fixture(u64_t seed) {
        splitmix64_t random_generator {seed};
        uniform_below<u32_t> const symbol {4};
        for (std::size_t entry = 0; entry != arrow_entries_k; ++entry) {
            for (std::size_t byte = 0; byte != arrow_sequence_bytes_k; ++byte)
                symbols += "ACGT"[symbol(random_generator)];
            offsets.push_back(symbols.size());
        }
    }
    usearch_tape_view_t values() const noexcept { return tape_of(symbols.data(), offsets.data(), usearch_scalar_u8_k); }
    usearch_tape_view_t queries() const noexcept { return values(); }
};

template <>
struct arrow_fixture<shapes_verbs_t> {
    std::vector<f32_t> coordinates;
    std::vector<std::size_t> offsets {0};
    usearch_shapes_metadata_t geometry {usearch_shapes_metric_chamfer_k, usearch_scalar_f32_k, 0, 0};

    explicit arrow_fixture(u64_t seed) {
        splitmix64_t random_generator {seed};
        uniform_between<f64_t> const unit {-1, 1};
        coordinates.resize(arrow_entries_k * arrow_cloud_points_k * 3);
        for (f32_t& coordinate : coordinates) coordinate = static_cast<f32_t>(unit(random_generator));
        for (std::size_t entry = 0; entry != arrow_entries_k; ++entry)
            offsets.push_back(offsets.back() + arrow_cloud_points_k);
    }
    usearch_tape_view_t values() const noexcept {
        return tape_of(coordinates.data(), offsets.data(), usearch_scalar_f32_k);
    }
    usearch_tape_view_t queries() const noexcept { return values(); }
};

template <>
struct arrow_fixture<geospatial_verbs_t> {
    std::vector<usearch_geospatial_observation_t> points;
    geospatial_geometry_t geometry {};

    explicit arrow_fixture(u64_t seed) : points(arrow_entries_k) {
        constexpr f64_t nan_k = std::numeric_limits<f64_t>::quiet_NaN();
        splitmix64_t random_generator {seed};
        uniform_between<f64_t> const across {-1, 1};
        for (usearch_geospatial_observation_t& point : points)
            point = {51.5 + across(random_generator) * 0.0045,
                     -0.1 + across(random_generator) * 0.0072,
                     nan_k,
                     nan_k,
                     nan_k,
                     nan_k,
                     nan_k,
                     nan_k,
                     nan_k,
                     nan_k,
                     nan_k};
    }
    usearch_geospatial_observation_t const* values() const noexcept { return points.data(); }
    geospatial_queries_t queries() const noexcept { return {points.data(), usearch_geospatial_relation_near_k, {}}; }
};

/** The directory and name every file of the case keyed by @p name starts with. */
static std::string tools_stem(std::string_view name, std::uint32_t seed) {
    char const* const directory = std::getenv("TMPDIR");
    return fmt::format("{}/usearch-tools-{}-{:08x}", directory ? directory : ".", name, seed);
}

/** The file at @p arrow_path holds one record batch per part and the keys as the UUID extension,
 *  and a dense one its vectors as fixed-size lists of bf16 elements. */
template <typename verbs_type_>
static std::size_t expect_arrow_schema(std::string const& arrow_path) {
    auto [mapping, mapping_status] = mapped_file_t::read(arrow_path);
    if (failed(mapping_status)) return check(false);
    std::array<ipc_field_t, ipc_fields_max_k> fields {};
    std::array<ipc_key_value_t, 4 * ipc_fields_max_k> pairs {};
    std::vector<ipc_block_t> blocks(arrow_parts_k);
    ipc_schema_t schema;
    std::size_t blocks_count = 0;
    ipc_status_t const status = ipc_read_footer(mapping.view(), fields, pairs, schema, blocks, blocks_count);
    std::size_t failures = check(status == ipc_status_t::success_k);
    if (failures) return failures;
    failures += check(blocks_count == arrow_parts_k && schema.fields.size() >= 3);
    ipc_field_t const& key = schema.fields[0];
    ipc_field_t const& value = schema.fields[1];
    failures += check(key.name == "key" && key.type == ipc_type_t::fixed_size_binary_k && key.width == 16 &&
                      arrow_metadata_value(key.metadata, ipc_extension_name_key_k) == ipc_extension_uuid_k);
    if constexpr (std::is_same_v<verbs_type_, dense_verbs_t>)
        failures += check(value.name == "value" && value.type == ipc_type_t::fixed_size_list_k &&
                          value.width == arrow_dimensions_k &&
                          arrow_metadata_value(value.metadata, arrow_dtype_key_k) == "bf16");
    return failures + check(value.name == "value" && schema.fields[2].name == "neighbors");
}

/** The matches of @p found, keyed by query number, read back from Arrow alike in either framing. */
static std::size_t expect_matches_round_trip(std::string const& stem, copied_matches_t const& found) {
    std::vector<usearch_key_t> queries;
    for (std::size_t query = 0; query != found.counts.size(); ++query)
        queries.push_back(std::bit_cast<usearch_key_t>(uuid_of(query)));
    std::size_t failures = 0;
    for (std::string_view const extension : {"arrow", "arrows"}) {
        std::string const path = fmt::format("{}-matches.{}", stem, extension);
        arrow_matches_t read;
        std::string failure = arrow_matches_write(path, queries, found);
        if (failure.empty()) failure = arrow_matches_read(path, read);
        if (!failure.empty()) fmt::println(stderr, "  {}", failure);
        failures += check(failure.empty() && read.matches == found && read.queries.size() == queries.size() &&
                          read.queries.back() == uuid_of(queries.size() - 1));
    }
    return failures;
}

/** Saves, exports, imports and reloads an index of two shards, as the file's docblock says. */
template <typename verbs_type_>
static std::size_t check_arrow_round_trip(settings_t const& settings) {
    session_t session = open_session(settings, settings.cpu_class);
    if (!session) return 1;
    arrow_fixture<verbs_type_> const fixture {settings.seed};
    handle<verbs_type_> index = open_index<verbs_type_>(session, fixture.geometry, arrow_parts_k);
    if (!index) return 1;
    usearch_traversal_stats_t stats {};
    std::size_t failures = check(insert_entries(index, fixture.values(), arrow_entries_k, 0, cell_batch_k, 0, stats));
    if (failures) return failures;

    std::string const stem = tools_stem(fmt::format("arrow-{}", verbs_type_::name_k), settings.seed);
    std::string const saved = fmt::format("{}.safegraphs", stem), arrow_path = fmt::format("{}.arrow", stem);
    std::string const resaved_stem = fmt::format("{}-imported", stem);
    usearch_outcome_t outcome {};
    verbs_type_::save(index.get(), index.executor(), bytes_view(saved), &outcome);
    failures += check(outcome);

    arrow_summary_t exported, imported;
    std::string checkpoint;
    std::string failure = arrow_export(saved, arrow_path, exported);
    if (failure.empty()) failure = arrow_import(arrow_path, "", checkpoint, imported);
    if (!failure.empty()) fmt::println(stderr, "  {}", failure);
    failures += check(failure.empty() && exported.parts == arrow_parts_k && exported.nodes == arrow_entries_k);
    failures += check(imported.parts == arrow_parts_k && imported.nodes == arrow_entries_k &&
                      imported.metric == exported.metric);
    failures += expect_arrow_schema<verbs_type_>(arrow_path);

    std::vector<std::int32_t> placement;
    typename verbs_type_::index_t copy = nullptr;
    verbs_type_::load_ram_buffer(&copy, session.executor, bytes_view(checkpoint), session.ram_options(0, placement),
                                 fixture.geometry, &outcome);
    failures += check(outcome);
    handle<verbs_type_> reloaded {copy, session, usearch_residency_ram_k, fixture.geometry};
    if (reloaded) {
        typename verbs_type_::description_t const original = describe(index), restored = describe(reloaded);
        typename verbs_type_::metadata_t const& before = original.metadata;
        typename verbs_type_::metadata_t const& after = restored.metadata;
        failures += check(restored.size == original.size && restored.shards_count == original.shards_count);
        failures += check(after.connectivity == before.connectivity);
        if constexpr (std::is_same_v<verbs_type_, dense_verbs_t>)
            failures += check(after.metric == before.metric && after.quantization == before.quantization &&
                              after.dimensions == before.dimensions);
        // Walks draw landmarks per thread, so only exact scans answer alike from run to run.
        auto const search = [&](handle<verbs_type_>& searched) {
            copied_matches_t found;
            usearch_traversal_stats_t searched_stats {};
            failures += check(search_queries(searched, fixture.queries(), arrow_queries_k, cell_batch_k, cell_wanted_k,
                                             0, usearch_traversal_scan_k, searched_stats, found));
            return found;
        };
        copied_matches_t const found = search(index);
        failures += check(equivalent_matches(search(reloaded), found, 0));
        failures += expect_matches_round_trip(stem, found);
        // The parts saved from the imported copy hold the same bytes as the original's, but for
        // shapes, whose load packs its poses again and whose save rounds them back.
        std::string const resaved_path = fmt::format("{}.safegraphs", resaved_stem);
        verbs_type_::save(reloaded.get(), reloaded.executor(), bytes_view(resaved_path), &outcome);
        failures += check(outcome);
        growing_buffer<char> original_path {global_allocator}, resaved_part {global_allocator};
        for (std::size_t part = 0; part != arrow_parts_k && !std::is_same_v<verbs_type_, shapes_verbs_t>; ++part) {
            auto const [first, first_status] = safegraphs_part_path(original_path, stem, part, arrow_parts_k);
            auto const [second, second_status] = safegraphs_part_path(resaved_part, resaved_stem, part, arrow_parts_k);
            failures += check(succeeded(first_status) && succeeded(second_status) &&
                              same_file_bytes(std::string {first}, std::string {second}));
        }
    }
    else failures += 1;
    for (std::string const& file : files_starting_with(stem)) std::remove(file.c_str());
    return failures;
}

/** Reports the round trip of the modality @p verbs_type_ names, when the filter selects it. */
template <typename verbs_type_>
static std::size_t report_arrow_round_trip(settings_t const& settings) {
    std::string const name = fmt::format("arrow round trip {}", verbs_type_::name_k);
    return settings.filter.selects(name) ? report(name.c_str(), check_arrow_round_trip<verbs_type_>(settings)) : 0;
}

/** A file standing as a drive of three slots maps as @c usearch @c slots prints it: the checkpoint
 *  in slot 0, the zeros of slot 1 free, and the bytes of slot 2 foreign, which every placement
 *  refuses like a live file. Saves bound themselves to slots on block devices alone, so one to this
 *  file is admitted as given. */
static std::size_t check_slots_skip_foreign(settings_t const& settings) {
    session_t session = open_session(settings, settings.cpu_class);
    if (!session) return 1;
    arrow_fixture<dense_verbs_t> const fixture {settings.seed};
    handle<dense_verbs_t> index = open_index<dense_verbs_t>(session, fixture.geometry, 1);
    if (!index) return 1;
    usearch_traversal_stats_t stats {};
    std::size_t failures = check(insert_entries(index, fixture.values(), arrow_entries_k, 0, cell_batch_k, 0, stats));
    if (failures) return failures;

    std::string const stem = tools_stem("slots", settings.seed);
    std::string const saved = fmt::format("{}.safegraphs", stem), location = fmt::format("{}.drive", stem);
    usearch_outcome_t outcome {};
    dense_verbs_t::save(index.get(), index.executor(), bytes_view(saved), &outcome);
    failures += check(outcome);
    std::error_code sized;
    std::uintmax_t const monolith_bytes = std::filesystem::file_size(saved, sized);
    failures += check(!sized && slot_copy(saved, location, 0, monolith_bytes));
    auto [foreign, foreign_status] = mapped_file_t::write(fmt::format("{}@{}", location, 2 * slot_bytes_k),
                                                          slot_probe_bytes_k);
    if (succeeded(foreign_status)) std::memset(foreign.span().data(), 0x5a, slot_probe_bytes_k);
    failures += check(succeeded(foreign_status) && succeeded(foreign.sync()));
    std::filesystem::resize_file(location, 3 * slot_bytes_k, sized);
    failures += check(!sized);

    slot_drive_t drive;
    std::vector<slot_file_t> files;
    failures += check(slot_open(location, drive) && slot_map(drive, location, files));
    failures += check(files.size() == 2);
    if (files.size() == 2) {
        failures += check(files[0].first_slot == 0 && files[0].owner != slot_foreign_k);
        failures += check(files[1].first_slot == 2 && files[1].owner == slot_foreign_k);
        failures += check(slot_overlap(files, 0, 1) == &files[0] && !slot_overlap(files, 1, 1) &&
                          slot_overlap(files, 1, 2) == &files[1]);
        failures += check(slot_free_run(files, 3, 1) == 1 && slot_free_run(files, 3, 2) == 3);
    }
    std::string const address = fmt::format("{}@1GiB", location);
    failures += check(checkpoints_admit_save(address) == address);
    for (std::string const& file : files_starting_with(stem)) std::remove(file.c_str());
    return failures;
}

#if USEARCH_OS_WINDOWS_

/** Points standard input at a pipe a child process fills from @p path, for mode @c r, or standard
 *  output at one it drains into @p path, for mode @c w, until destroyed, which closes the pipe and
 *  waits for the child. */
struct piped_descriptor_t {
    DWORD stream;
    native_file_t saved;
    native_file_t ours = invalid_native_file_k;
    PROCESS_INFORMATION child {};

    piped_descriptor_t(char const* mode, std::string const& path)
        : stream(*mode == 'r' ? STD_INPUT_HANDLE : STD_OUTPUT_HANDLE), saved(::GetStdHandle(stream)) {
        SECURITY_ATTRIBUTES inherited {sizeof(SECURITY_ATTRIBUTES), nullptr, TRUE};
        native_file_t read_end = invalid_native_file_k, write_end = invalid_native_file_k;
        if (!::CreatePipe(&read_end, &write_end, &inherited, 0)) return;
        bool const reading = stream == STD_INPUT_HANDLE;
        ours = reading ? read_end : write_end;
        native_file_t const theirs = reading ? write_end : read_end;
        ::SetHandleInformation(ours, HANDLE_FLAG_INHERIT, 0);
        STARTUPINFOA startup {};
        startup.cb = sizeof(startup);
        startup.dwFlags = STARTF_USESTDHANDLES;
        startup.hStdInput = reading ? nullptr : theirs;
        startup.hStdOutput = reading ? theirs : ::GetStdHandle(STD_ERROR_HANDLE);
        startup.hStdError = ::GetStdHandle(STD_ERROR_HANDLE);
        // `type` copies a file byte for byte, and Windows ships no such copier from standard input.
        std::string command = reading ? fmt::format("cmd.exe /d /c type \"{}\"", path)
                                      : fmt::format("powershell.exe -NoProfile -NonInteractive -Command \"$file = " //
                                                    "[IO.File]::Create('{}'); [Console]::OpenStandardInput()."      //
                                                    "CopyTo($file); $file.Close()\"",
                                                    path);
        if (!::CreateProcessA(nullptr, command.data(), nullptr, nullptr, TRUE, 0, nullptr, nullptr, &startup, &child))
            child = {};
        ::CloseHandle(theirs);
        ::SetStdHandle(stream, ours);
    }
    piped_descriptor_t(piped_descriptor_t const&) = delete;
    piped_descriptor_t& operator=(piped_descriptor_t const&) = delete;
    ~piped_descriptor_t() noexcept {
        ::SetStdHandle(stream, saved);
        if (ours != invalid_native_file_k) ::CloseHandle(ours);
        if (!child.hProcess) return;
        ::WaitForSingleObject(child.hProcess, INFINITE);
        ::CloseHandle(child.hThread);
        ::CloseHandle(child.hProcess);
    }
};

#else

/** Points standard input at a pipe @c cat fills from @p path, for mode @c r, or standard output at
 *  one @c cat drains into @p path, for mode @c w, until destroyed, which closes the pipe and waits
 *  for the command. */
struct piped_descriptor_t {
    native_file_t descriptor;
    native_file_t saved;
    std::FILE* pipe;

    piped_descriptor_t(char const* mode, std::string const& path)
        : descriptor(*mode == 'r' ? standard_input_native_file() : standard_output_native_file()),
          saved(dup(descriptor)), pipe(popen(fmt::format("cat {}'{}'", *mode == 'r' ? "" : "> ", path).c_str(), mode)) {
        // What the suite printed so far stays out of the pipe.
        std::fflush(stdout);
        dup2(fileno(pipe), descriptor);
    }
    piped_descriptor_t(piped_descriptor_t const&) = delete;
    piped_descriptor_t& operator=(piped_descriptor_t const&) = delete;
    ~piped_descriptor_t() noexcept {
        dup2(saved, descriptor);
        close(saved);
        pclose(pipe);
    }
};

#endif

/** Rows of the columns a foreign producer writes: its own integer IDs, and its embeddings, packed
 *  as Arrow lays out a fixed-size list. */
struct arrow_table_t {
    std::vector<std::int64_t> identifiers;
    std::vector<f32_t> embeddings;

    explicit arrow_table_t(u64_t seed) {
        nk::matrix<nk::f32_t> const vectors = make_vectors<nk::f32_t>(arrow_entries_k, arrow_dimensions_k, seed);
        embeddings.resize(arrow_entries_k * arrow_dimensions_k);
        for (std::size_t entry = 0; entry != arrow_entries_k; ++entry) {
            identifiers.push_back(static_cast<std::int64_t>(entry) * 7 + 3);
            std::memcpy(embeddings.data() + entry * arrow_dimensions_k,
                        reinterpret_cast<char const*>(vectors.data()) + entry * vectors.stride_bytes(0),
                        arrow_dimensions_k * sizeof(f32_t));
        }
        for (f32_t& value : embeddings) value = std::bit_cast<f32_t>(std::bit_cast<std::uint32_t>(value) & 0xFFFF0000u);
    }

    std::string write(std::string_view path) const {
        std::int64_t const rows = static_cast<std::int64_t>(identifiers.size());
        ipc_field_t const element {.name = arrow_child_name_k, .type = ipc_type_t::floating_point_k, .width = 32};
        std::array<ipc_field_t, 2> const fields {{
            {.name = "id", .type = ipc_type_t::int_k, .width = 64, .signedness = ipc_signedness_t::signed_k},
            {.name = "embedding",
             .type = ipc_type_t::fixed_size_list_k,
             .width = arrow_dimensions_k,
             .children = {&element, 1}},
        }};
        std::int64_t const elements = rows * static_cast<std::int64_t>(arrow_dimensions_k);
        std::array<ipc_node_t, 3> const nodes {{{rows, 0}, {rows, 0}, {elements, 0}}};
        std::array<bytes_view_t, 5> const buffers {
            bytes_view_t {}, std::as_bytes(std::span {identifiers}), bytes_view_t {},
            bytes_view_t {}, std::as_bytes(std::span {embeddings}),
        };
        return arrow_write_batch(path, {fields, {}}, {rows, nodes, buffers});
    }
};

/** Whether @p input holds exactly @p table through the converters, its integer IDs widened, and its
 *  rows picked for being the only ones. */
static std::size_t expect_arrow_table(arrow_input_t& input, arrow_table_t const& table) {
    std::size_t vectors = arrow_no_column_k, identifiers = arrow_no_column_k;
    std::size_t failures = check(arrow_pick_rows(input, "", vectors).empty() &&
                                 arrow_pick_column(input, "id", identifiers).empty());
    if (failures) return failures;
    bool more = false;
    failures += check(input.next(more).empty() && more && input.rows() == arrow_entries_k);
    if (failures) return failures;
    usearch_rows_view_t rows {};
    usearch_keys_view_t keys {};
    std::vector<usearch_key_t> widened;
    failures += check(arrow_rows_of(input, vectors, rows).empty() &&
                      arrow_keys_of(input, identifiers, widened, keys).empty());
    if (failures) return failures;
    failures += check(rows.columns == arrow_dimensions_k && rows.scalar == usearch_scalar_f32_k &&
                      std::memcmp(rows.data, table.embeddings.data(), table.embeddings.size() * sizeof(f32_t)) == 0);
    failures += check(std::bit_cast<uuid128_t>(keys.data[1]) == uuid_of(static_cast<u64_t>(table.identifiers[1])));
    failures += check(input.next(more).empty() && !more);
    return failures;
}

/** Arrow crosses real pipes on the standard descriptors in either framing, a stream cut before its
 *  end refuses instead of reading short, and Parquet refuses by name. */
static std::size_t check_arrow_pipes(settings_t const& settings) {
    arrow_table_t const table {settings.seed};
    std::string const stem = tools_stem("pipes", settings.seed);
    std::size_t failures = 0;
    for (std::string_view const extension : {"arrow", "arrows"}) {
        std::string const path = fmt::format("{}-table.{}", stem, extension);
        failures += check(table.write(path).empty());
        piped_descriptor_t const piped {"r", path};
        arrow_input_t input;
        failures += check(input.open("-").empty());
        failures += expect_arrow_table(input, table);
    }

    copied_matches_t found;
    found.counts = {1, 0};
    found.keys = {uuid_of(5)};
    found.scores = {0.25};
    std::array<usearch_key_t, 2> const queries {std::bit_cast<usearch_key_t>(uuid_of(0)),
                                                std::bit_cast<usearch_key_t>(uuid_of(1))};
    std::string const piped_path = fmt::format("{}-piped.arrows", stem);
    {
        piped_descriptor_t const piped {"w", piped_path};
        failures += check(arrow_matches_write("-", queries, found).empty());
    }
    arrow_matches_t read;
    failures += check(arrow_matches_read(piped_path, read).empty() && read.matches == found);

    std::string const cut = fmt::format("{}-cut.arrows", stem);
    failures += check(table.write(cut).empty());
    std::error_code sized;
    std::filesystem::resize_file(cut, std::filesystem::file_size(cut, sized) - sizeof(std::uint64_t), sized);
    arrow_input_t truncated;
    bool more = false;
    failures += check(!sized && truncated.open(cut).empty() && truncated.next(more).empty() && more);
    failures += check(!truncated.next(more).empty());

    std::string const parquet = fmt::format("{}.parquet", stem);
    std::FILE* const file = std::fopen(parquet.c_str(), "wb");
    failures += check(file && std::fputs("PAR1 and no more", file) >= 0 && std::fclose(file) == 0);
    arrow_input_t refused;
    failures += check(refused.open(parquet).find("Parquet") != std::string::npos);
    for (std::string const& leftover : files_starting_with(stem)) std::remove(leftover.c_str());
    return failures;
}

/** The built @p binary chains through pipes as a shell would: an insert from a stream into a pipe,
 *  whose status lines leave standard output to the checkpoint, the metadata of that checkpoint, an
 *  exact search reading it off a pipe and answering every row's own key, and a get answering rows
 *  byte for byte. */
static std::size_t check_tool_pipes(settings_t const& settings, std::string_view binary) {
    arrow_table_t const table {settings.seed};
    std::string const stem = tools_stem("tool", settings.seed);
    std::string const stream = fmt::format("{}.arrows", stem), index = fmt::format("{}.safegraphs", stem);
    std::string const run = fmt::format("{}-run.arrows", stem), vectors = fmt::format("{}-vectors.arrow", stem);
    std::size_t failures = check(table.write(stream).empty());
    auto const run_shell = [&](std::string const& command) {
        return check(std::system(command.c_str()) == EXIT_SUCCESS);
    };
    failures += run_shell(
        fmt::format("cat '{}' | '{}' dense insert --input - --keys-column id --stored-dtype bf16 --save - > '{}'",
                    stream, binary, index));
    failures += run_shell(fmt::format("'{}' metadata - < '{}' > /dev/null", binary, index));
    failures += run_shell(fmt::format("cat '{}' | '{}' dense search --index - --input '{}' --keys-column id " //
                                      "--traversal scan --wanted 1 --output - > '{}'",                        //
                                      index, binary, stream, run));
    arrow_matches_t searched;
    failures += check(arrow_matches_read(run, searched).empty() && searched.queries.size() == arrow_entries_k);
    if (!failures) failures += check(searched.matches.keys == searched.queries);

    failures += run_shell(
        fmt::format("'{}' dense get --index '{}' --input '{}' --keys-column id --outputs-dtype f32 --output '{}'",
                    binary, index, stream, vectors));
    arrow_input_t retrieved;
    std::size_t rows_column = arrow_no_column_k;
    bool more = false;
    usearch_rows_view_t rows {};
    failures += check(retrieved.open(vectors).empty() && arrow_pick_column(retrieved, "vector", rows_column).empty());
    failures += check(retrieved.next(more).empty() && more && arrow_rows_of(retrieved, rows_column, rows).empty());
    if (!failures)
        failures += check(std::memcmp(rows.data, table.embeddings.data(), table.embeddings.size() * sizeof(f32_t)) ==
                          0);
    for (std::string const& leftover : files_starting_with(stem)) std::remove(leftover.c_str());
    return failures;
}

} // namespace unum::usearch::test::abi

using namespace unum::usearch::test;

int main(int argc, char** argv) {
    settings_t const settings = read_settings(argc, argv, "usearch_test_tools");
    std::size_t failures_count = 0;
    failures_count += abi::report_arrow_round_trip<abi::dense_verbs_t>(settings);
    failures_count += abi::report_arrow_round_trip<abi::sequence_verbs_t>(settings);
    failures_count += abi::report_arrow_round_trip<abi::shapes_verbs_t>(settings);
    failures_count += abi::report_arrow_round_trip<abi::geospatial_verbs_t>(settings);
    if (settings.filter.selects("slots skip foreign"))
        failures_count += report("slots skip foreign", abi::check_slots_skip_foreign(settings));
    if (settings.filter.selects("arrow pipes"))
        failures_count += report("arrow pipes", abi::check_arrow_pipes(settings));
    char const* const binary = std::getenv("USEARCH_BINARY");
    if (binary && settings.filter.selects("tool pipes"))
        failures_count += report("tool pipes", abi::check_tool_pipes(settings, binary));
    return report("total", failures_count) ? 1 : 0;
}
