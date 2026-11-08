/**
 *  @file tools/slots.hpp
 *  @author Ash Vardanian
 *  @date October 18, 2026
 *  @brief The slots of a raw drive: mapping the live files on it, admitting a save to a free run,
 *      and verified copies of a part onto a slot, which the tools and the benchmarks share.
 *
 *  A raw drive is cut into slots of 1 GiB from byte 0, and every file or volume on it starts on a
 *  slot boundary and covers whole slots; slot 0 holds no label. Mapping reads the first 4 KiB of
 *  each slot, which is live when it opens a SafeGraphs monolith whose header parses and whose
 *  declared extent fits the drive, free when it is all zeros, and foreign otherwise. A live file
 *  covers ⌈length ÷ 1 GiB⌉ slots, which mapping skips, and a foreign slot is skipped unidentified
 *  and never written. Owners come from the SafeGraphs metadata.
 *
 *  Every transfer goes through the library's own files, so the helpers build wherever the harness
 *  does: positional reads through @c file_medium_t, which reaches a generic char node as a view
 *  does, and writes through @c mapped_file_t, which opens a disk exclusively and writes it direct
 *  from anonymous memory.
 */
#pragma once

#include <cstdio>  // `stderr`
#include <cstring> // `std::memcmp`, `std::memcpy`, `std::memset`

#include <chrono>       // `std::chrono::steady_clock` - the copy rates
#include <filesystem>   // `std::filesystem::is_block_file`, `is_character_file`, `resize_file` - drives, staged parts
#include <iterator>     // `std::back_inserter`
#include <memory>       // `std::allocator`
#include <optional>     // `std::optional`
#include <span>         // `std::span`
#include <string>       // `std::string`
#include <string_view>  // `std::string_view`
#include <system_error> // `std::error_code`
#include <vector>       // `std::vector`

#include <fmt/format.h>

#include "usearch/utilities/blocking_files.hpp"    // `mapped_file_t`, `parse_file_address`
#include "usearch/utilities/nvme_device.hpp"       // `file_medium_t`, `medium_bytes_of`, `names_transport_id`
#include "usearch/utilities/safegraphs_format.hpp" // `safegraphs_read_header`, `safegraphs_read_manifest`
#include "usearch/utilities/types.hpp"             // `divide_round_up`, `little_endian`, `u64_t`

namespace unum::usearch::tools {
using namespace unum::usearch::utilities;

/** Bytes in one slot of a raw drive, where every file or volume starts and whose multiples it
 *  covers. */
inline constexpr u64_t slot_bytes_k = 1ull << 30;

/** Bytes mapping reads from the front of every slot. */
inline constexpr std::size_t slot_probe_bytes_k = 4096;

/** Bytes one staged write of a copy moves. */
inline constexpr std::size_t copy_chunk_bytes_k = 16ull << 20;

/** The owner of a slot that is neither free nor ours, which mapping skips and a save never
 *  overwrites. */
inline constexpr std::string_view slot_foreign_k = "foreign";

/** A medium holding slots, open for positional reads, and the bytes it spans. */
struct slot_drive_t {
    file_medium_t file;
    u64_t bytes = 0;
};

/** A live file on a drive: its first slot, the bytes its header declares, and who owns it. */
struct slot_file_t {
    u64_t first_slot = 0;
    u64_t bytes = 0;
    std::string owner;
};

/** One part a placement copies: the file or slot holding it, and its bytes. */
struct staged_part_t {
    std::string address;
    u64_t bytes = 0;
};

/** Slots a live file of @p bytes covers, at least the one it starts in. */
inline u64_t slots_covered(u64_t bytes) noexcept {
    return max_of<u64_t>(1, divide_round_up<u64_t>(bytes, slot_bytes_k));
}

/** The noun a count of @p slots takes. */
inline std::string_view slots_noun(u64_t slots) noexcept { return slots == 1 ? "slot" : "slots"; }

/** The little-endian integer at byte @p offset of @p bytes. */
template <typename integer_type_>
integer_type_ slot_integer_at(bytes_view_t bytes, std::size_t offset) noexcept {
    integer_type_ value = 0;
    std::memcpy(&value, bytes.data() + offset, sizeof(value));
    return little_endian(value);
}

/** Opens the medium at @p location for reading and measures it; false after printing why. */
inline bool slot_open(std::string_view location, slot_drive_t& drive) {
    auto [file, open_status] = file_medium_t::open(location);
    if (failed(open_status)) {
        fmt::println(stderr, "cannot open {}", location);
        return false;
    }
    drive.bytes = medium_bytes_of(file), drive.file = std::move(file);
    return true;
}

/** Whether @p location names a block device. */
inline bool slot_names_device(std::string_view location) {
    std::error_code error;
    return std::filesystem::is_block_file(std::filesystem::path {location}, error);
}

/** Reads the SafeGraphs monolith opening at byte @p offset of @p drive, whose first bytes are
 *  @p probe, into @p file; false unless its header parses and its extent fits the drive. */
inline bool slot_probe_safegraphs(slot_drive_t const& drive, u64_t offset, bytes_view_t probe, slot_file_t& file) {
    using prefix_t = safegraphs_prefix_t;
    if (std::memcmp(probe.data(), prefix_t::magic_k.data(), prefix_t::magic_k.size()) != 0) return false;
    u64_t const header_length = slot_integer_at<u64_t>(probe, prefix_t::magic_k.size());
    if (header_length > drive.bytes - offset - sizeof(prefix_t)) return false;
    std::size_t const header_bytes = static_cast<std::size_t>(sizeof(prefix_t) + header_length);
    std::vector<byte_t> longer;
    bytes_view_t header = probe.first(min_of(header_bytes, probe.size()));
    if (header_bytes > probe.size()) {
        longer.resize(header_bytes);
        if (failed(drive.file.read_at(offset, longer))) return false;
        header = longer;
    }
    std::string metric, producer;
    auto per_custom_field = [&](safegraphs_custom_field_t field) noexcept {
        if (field.key == "metric") metric = field.value;
        else if (field.key == "created_by") producer = field.value;
    };
    // Producer columns such as the norms extend the monolith, so the header lists them too.
    growing_buffer<safegraphs_named_column_t, std::allocator<byte_t>> extra_columns {std::allocator<byte_t> {}};
    auto const [parsed, status] = safegraphs_read_header(header, std::allocator<byte_t> {}, per_custom_field,
                                                         &extra_columns);
    if (failed(status) || parsed.monolith_bytes() > drive.bytes - offset) return false;
    file.first_slot = offset / slot_bytes_k;
    file.bytes = parsed.monolith_bytes();
    file.owner = fmt::format("SafeGraphs of {} nodes of {}, metric \"{}\"", parsed.nodes(),
                             dtype_name(parsed.value.dtype), metric);
    if (!producer.empty()) fmt::format_to(std::back_inserter(file.owner), ", created by \"{}\"", producer);
    return true;
}

/** Classifies the slot at byte @p offset of @p drive, whose first bytes are @p probe, into @p file;
 *  false where its first page is all zeros. A slot that is not ours is one slot long and owned by
 *  @c slot_foreign_k, so overlap and free-run searches refuse it like any live file. */
inline bool slot_classify(slot_drive_t const& drive, u64_t offset, bytes_view_t probe, slot_file_t& file) {
    if (slot_probe_safegraphs(drive, offset, probe, file)) return true;
    if (probe[0] == byte_t {} && !std::memcmp(probe.data(), probe.data() + 1, probe.size() - 1)) return false;
    file.first_slot = offset / slot_bytes_k, file.bytes = slot_bytes_k, file.owner = slot_foreign_k;
    return true;
}

/** Reads the file or foreign slot opening at byte @p offset of @p drive into @p file; false where
 *  the slot is free. */
inline bool slot_probe(slot_drive_t const& drive, u64_t offset, slot_file_t& file) {
    std::vector<byte_t> probe(slot_probe_bytes_k);
    if (probe.size() > drive.bytes - min_of(offset, drive.bytes) || failed(drive.file.read_at(offset, probe)))
        return false;
    return slot_classify(drive, offset, probe, file);
}

/** Maps every live and foreign slot on @p drive into @p files in slot order; false after printing
 *  a failed read. */
inline bool slot_map(slot_drive_t const& drive, std::string_view location, std::vector<slot_file_t>& files) {
    u64_t const slots = drive.bytes / slot_bytes_k;
    std::vector<byte_t> probe(slot_probe_bytes_k);
    for (u64_t slot = 0; slot < slots;) {
        if (failed(drive.file.read_at(slot * slot_bytes_k, probe))) {
            fmt::println(stderr, "cannot read slot {} of {}", slot, location);
            return false;
        }
        slot_file_t file;
        if (!slot_classify(drive, slot * slot_bytes_k, probe, file)) {
            ++slot;
            continue;
        }
        slot += slots_covered(file.bytes);
        files.push_back(std::move(file));
    }
    return true;
}

/** The live file covering a slot of @p count from @p first, or null where they are all free. */
inline slot_file_t const* slot_overlap(std::span<slot_file_t const> files, u64_t first, u64_t count) noexcept {
    for (slot_file_t const& file : files)
        if (file.first_slot < first + count && first < file.first_slot + slots_covered(file.bytes)) return &file;
    return nullptr;
}

/** The first slot of the first free run of @p count among @p slots, or @p slots where none is long
 *  enough. */
inline u64_t slot_free_run(std::span<slot_file_t const> files, u64_t slots, u64_t count) noexcept {
    u64_t start = 0;
    for (slot_file_t const& file : files) {
        if (file.first_slot >= start + count) break;
        start = max_of(start, file.first_slot + slots_covered(file.bytes));
    }
    return start + count <= slots ? start : slots;
}

/** The slot ending the free run that starts at the free slot @p first: the next live file's first,
 *  or @p slots. */
inline u64_t slot_run_end(std::span<slot_file_t const> files, u64_t slots, u64_t first) noexcept {
    for (slot_file_t const& file : files)
        if (file.first_slot > first) return min_of(file.first_slot, slots);
    return slots;
}

/** Opens the drive @p address names and maps it, refusing an address that does not parse. */
inline bool slot_open_drive(std::string_view address, std::string& location, slot_drive_t& drive,
                            std::vector<slot_file_t>& files) {
    auto const [parsed, status] = parse_file_address(address);
    if (failed(status)) {
        fmt::println(stderr, "{} is no address", address);
        return false;
    }
    location = parsed.location;
    return slot_open(location, drive) && slot_map(drive, location, files);
}

/** The address a save may write to at @p address, printing why none: @c - for standard output, a
 *  file or a transport ID as given, and a drive bounded to the free run its slot starts, since
 *  nothing sizes a save before it writes. A drive takes a byte base on a slot boundary, at a slot
 *  no live file covers. */
inline std::optional<std::string> checkpoints_admit_save(std::string_view address) {
    if (address == "-") return std::string {address};
    // A drive bound to SPDK is out of the kernel's reach, and only the library maps it.
    if (names_transport_id(address)) return std::string {address};
    auto const [parsed, status] = parse_file_address(address);
    if (failed(status)) {
        fmt::println(stderr, "{} is no address", address);
        return std::nullopt;
    }
    std::string const location {parsed.location};
    std::error_code error;
    // A generic char node takes only NVMe commands, and the library's writers issue none.
    if (std::filesystem::is_character_file(std::filesystem::path {location}, error)) {
        fmt::println(stderr,
                     "{} is a generic char node, which takes no writes; save through its namespace's block " //
                     "node, like /dev/nvme1n1@64GiB, and read the slot back through {}",
                     address, location);
        return std::nullopt;
    }
    if (!slot_names_device(location)) return std::string {address};
    if (parsed.location.size() == address.size()) {
        fmt::println(stderr,
                     "{} names a drive without a byte base, and no call sizes a save before it writes; name a free " //
                     "slot that usearch slots lists, like {}@<slot>GiB",
                     address, location);
        return std::nullopt;
    }
    if (parsed.byte_base % slot_bytes_k) {
        fmt::println(stderr, "{} starts off a slot boundary, a multiple of 1 GiB", address);
        return std::nullopt;
    }
    slot_drive_t drive;
    std::vector<slot_file_t> files;
    if (!slot_open(location, drive) || !slot_map(drive, location, files)) return std::nullopt;
    u64_t const slots = drive.bytes / slot_bytes_k, first = parsed.byte_base / slot_bytes_k;
    if (first >= slots) {
        fmt::println(stderr, "{} starts past the {} slots of {}", address, slots, location);
        return std::nullopt;
    }
    if (slot_file_t const* live = slot_overlap(files, first, 1)) {
        fmt::println(stderr, "{} lands on {} at {}@{}GiB", address, live->owner, location, live->first_slot);
        return std::nullopt;
    }
    // The save may grow up to the next live file, and the library refuses a byte past it.
    u64_t const run = slot_run_end(files, slots, first) - first;
    if (parsed.byte_limit && parsed.byte_limit <= run * slot_bytes_k) return std::string {address};
    return fmt::format("{}@{}GiB+{}GiB", location, first, run);
}

/** Copies the @p length bytes at the address @p from onto @p location from byte @p base, a chunk
 *  at a time staged in memory a disk takes direct, then reads them back and compares; false after
 *  printing what failed. A disk opens exclusively, so a mounted one refuses the first chunk. */
inline bool slot_copy(std::string_view from, std::string_view location, u64_t base, u64_t length) {
    using steady_clock_t = std::chrono::steady_clock;
    auto [source, source_status] = mapped_file_t::read(from);
    if (failed(source_status) || source.size() < length) {
        fmt::println(stderr, "cannot map {} bytes of {}", length, from);
        return false;
    }
    bytes_view_t const copied = source.view().first(static_cast<std::size_t>(length));
    steady_clock_t::time_point started = steady_clock_t::now();
    for (u64_t offset = 0; offset < length; offset += copy_chunk_bytes_k) {
        std::size_t const chunk = static_cast<std::size_t>(min_of<u64_t>(copy_chunk_bytes_k, length - offset));
        auto [staged, staged_status] = mapped_file_t::write(fmt::format("{}@{}", location, base + offset), chunk);
        if (succeeded(staged_status)) std::memcpy(staged.span().data(), copied.data() + offset, chunk);
        if (failed(staged_status) || failed(staged.sync())) {
            fmt::println(stderr, "cannot write {} bytes at {} of {}", chunk, base + offset, location);
            return false;
        }
    }
    double const written_seconds = std::chrono::duration<double>(steady_clock_t::now() - started).count();
    fmt::println("copied {} to {} at byte {}: {} bytes in {:.1f} s, {:.2f} GB/s", from, location, base, length,
                 written_seconds, static_cast<double>(length) / written_seconds / 1e9);

    // A direct write drops the cached pages it covers, so the read back comes from the drive.
    started = steady_clock_t::now();
    auto [target, target_status] = mapped_file_t::read(fmt::format("{}@{}", location, base));
    if (failed(target_status) || target.size() < length) {
        fmt::println(stderr, "cannot read back {} bytes at {} of {}", length, base, location);
        return false;
    }
    for (u64_t offset = 0; offset < length; offset += copy_chunk_bytes_k) {
        std::size_t const chunk = static_cast<std::size_t>(min_of<u64_t>(copy_chunk_bytes_k, length - offset));
        if (std::memcmp(copied.data() + offset, target.view().data() + offset, chunk) != 0) {
            fmt::println(stderr, "{} differs from {} within the {} bytes at {}", location, from, chunk, offset);
            return false;
        }
    }
    fmt::println("verified {} bytes in {:.1f} s", length,
                 std::chrono::duration<double>(steady_clock_t::now() - started).count());
    return true;
}

/** Copies every part of the checkpoint at @p address onto @p paths, part @c i onto @c paths[i]
 *  admitted for its exact bytes, so a view relocating its parts onto @p paths serves them from the
 *  drives; false after printing why. Staged part files then drop their bytes, the first keeping its
 *  header, which a view reads the magic and a manifest reader the schema from. */
inline bool place_parts_on_slots(std::string_view address, std::span<std::string_view const> paths) {
    auto const [parsed, parse_status] = parse_file_address(address);
    slot_drive_t drive;
    if (failed(parse_status) || !slot_open(parsed.location, drive)) return false;
    std::vector<staged_part_t> parts;
    slot_file_t monolith;
    if (slot_probe(drive, parsed.byte_base, monolith) && monolith.owner != slot_foreign_k)
        parts.push_back({std::string {address}, monolith.bytes});
    else {
        std::string_view const directory = address.substr(0, address.size() - safegraphs_basename(address).size());
        auto per_part = [&](safegraphs_part_t part) noexcept {
            parts.push_back({fmt::format("{}{}", directory, part.path), part.bytes});
        };
        auto const [summary, summary_status] = safegraphs_read_manifest(address, std::allocator<byte_t> {}, per_part);
        if (failed(summary_status)) {
            fmt::println(stderr, "{} holds no SafeGraphs monolith or manifest", address);
            return false;
        }
    }
    if (parts.size() != paths.size()) {
        fmt::println(stderr, "{} holds {} parts, and USEARCH_NVME_PATHS names {} places", address, parts.size(),
                     paths.size());
        return false;
    }
    for (std::size_t part = 0; part != parts.size(); ++part) {
        std::optional<std::string> const admitted = checkpoints_admit_save(paths[part]);
        if (!admitted) return false;
        auto const [target, target_status] = parse_file_address(*admitted);
        if (names_transport_id(*admitted) || !slot_names_device(target.location)) {
            fmt::println(stderr, "{} is no block device; a part is written through one, like /dev/nvme1n1@64GiB",
                         paths[part]);
            return false;
        }
        if (target.byte_limit < parts[part].bytes) {
            fmt::println(stderr, "part {} of {} bytes outgrows the free run at {}", part, parts[part].bytes, *admitted);
            return false;
        }
        if (!slot_copy(parts[part].address, target.location, target.byte_base, parts[part].bytes)) return false;
    }
    std::error_code error;
    for (std::size_t part = 0; part != parts.size(); ++part) {
        std::filesystem::path const staged {parts[part].address};
        if (!std::filesystem::is_regular_file(staged, error)) continue;
        if (part) std::filesystem::remove(staged, error);
        else if (auto const [header, header_status] = safegraphs_read_header(parts[part].address);
                 succeeded(header_status))
            std::filesystem::resize_file(staged, header.data_offset, error);
    }
    return true;
}

} // namespace unum::usearch::tools
